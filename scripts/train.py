"""
Treina o classificador de risco de reencaminhamento de pedidos LAI.

O desenho decorre das auditorias registradas em docs/VERIFICATION.md. Cinco
famílias de variáveis foram excluídas por serem posteriores à triagem:

  * AssuntoPedido / SubAssuntoPedido são SAÍDAS da triagem (79,6% ausentes em
    pedidos de 3 dias, 0,000% após respondidos). Variante B mede o vazamento.
  * prazo_dias reimportava FoiProrrogado: mediana de 21 d sem prorrogação contra
    31 d com, exatamente os +10 d do art. 11 §2 da LAI. Variante C mede.
  * protocolo_seq codifica a unidade registradora, não um sequencial neutro:
    separação de 79x dentro de um mesmo órgão-ano (H5). Em quarentena.
  * OrgaoDestinatario é o órgão ENDEREÇADO, não o final -> mantido.
  * Linhas com Situacao == "Encaminhada por Outro Órgão" estão em trânsito ->
    descartadas.

Conjunto de produção (30 variáveis), após a segunda rodada de engenharia:
  14 categóricas + 6 numéricas de base
  + 2 de histórico agregado do solicitante          (G2, +0,0287 de PR-AUC)
  + 5 de experiência refinada do solicitante        (T1, +0,0434)
  + 2 de taxa móvel por órgão                       (+0,0056)
  + 1 de idade do órgão

As do solicitante são FORNECIDAS PELO CHAMADOR em produção, não embarcadas no
artefato — ver docs/DECISAO_ESTADO_SOLICITANTE.md. As do órgão são embarcadas.

Corte temporal, nunca aleatório. Métrica primária: precisão@k, não AUC, porque o
entregável é uma fila limitada pela capacidade de analistas seniores.

Procedimento reproduzível completo em docs/TREINAMENTO.md.
"""

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# P1: caminhos, constantes, limpeza e métrica vêm do pacote. Antes estavam
# duplicados aqui e em mais cinco scripts, e a correção do H9 alcançou só um
# deles. Reexportados como nomes de módulo porque `experiment_h7_h8.py` e
# `verify_h7_alinhamento.py` os acessam como `train.MATURITY_DAYS`.
from lai_triagem.codificacao import orgao_rate_crossfit, taxa_suavizada  # noqa: E402
from lai_triagem.config import (  # noqa: E402
    ART,
    COHORT,
    MATURITY_DAYS,
    PRIOR_MOVEL,  # noqa: F401  — reexportado: `verify_h7_alinhamento.py` e
    # `experiment_h7_h8.py` leem `train.PRIOR_MOVEL`. `BIRTH_YEARS`
    # saiu junto com `organ_birth_table` e ninguém o lia por aqui.
    PRIOR_ORGAO,
    QUEUE_FRAC,
    READ_KW,
    SEED,
    SNAPSHOT,
    TEST_YEAR,
    TRAIN_YEARS,
    VAL_YEAR,
)
from lai_triagem.config import (
    RAIZ as ROOT,
)
from lai_triagem.dados import arquivo_mais_recente, limpar  # noqa: E402
from lai_triagem.metricas import precision_at_k  # noqa: E402

# EXPERIMENTO TEMPORIAN. A construção de variáveis saiu daqui e foi para
# `lai_triagem/variaveis_temporian.py`, onde as janelas passam a ser operações
# do Temporian em vez de `searchsorted`/`merge_asof` à mão. O resto deste
# arquivo — corte temporal, variantes, métrica, artefato — não mudou.
from lai_triagem.variaveis_temporian import (  # noqa: E402
    build_features,
    organ_birth_table,
    taxas_moveis_por_orgao,
)

ART.mkdir(exist_ok=True)

# Apelidos dos importados: as chamadas internas continuam `_clean(...)` e
# `_latest(...)`, e `train._clean` segue valido para quem importa este modulo.
_clean = limpar
_latest = arquivo_mais_recente
# `organ_rolling` era função deste arquivo e virou `taxas_moveis_por_orgao` no
# módulo novo, com a mesma assinatura e o mesmo formato de retorno.
# `experiment_h7_h8.py`, `experiment_features_v2.py` e
# `verify_h7_alinhamento.py` a chamam pelo nome antigo. Atribuição, e não
# `import ... as`, porque o ruff apaga o segundo como importação sem uso — e
# apagar silenciosamente uma reexportação quebraria três scripts.
organ_rolling = taxas_moveis_por_orgao

PED_COLS = ["IdPedido", "ProtocoloPedido", "Esfera", "UF", "Municipio",
            "OrgaoDestinatario", "Situacao", "DataRegistro", "PrazoAtendimento",
            "FoiReencaminhado", "FormaResposta", "OrigemSolicitacao",
            "IdSolicitante", "AssuntoPedido", "SubAssuntoPedido"]
SOL_COLS = ["IdSolicitante", "TipoDemandante", "DataNascimento", "Genero", "Escolaridade",
            "Profissao", "TipoPessoaJuridica", "Pais", "UF", "Municipio"]

CAT_BASE = ["Esfera", "UF", "Municipio", "OrgaoDestinatario", "FormaResposta",
            "OrigemSolicitacao", "TipoDemandante", "Genero", "Escolaridade", "Profissao",
            "TipoPessoaJuridica", "Pais", "UF_sol", "Municipio_sol"]
NUM_BASE = ["reg_month", "reg_dow", "reg_day", "idade", "orgao_rate", "uf_match"]
# Variáveis de histórico que entram no modelo (dado pessoal, não embarcado).
NUM_SOLICITANTE = ["n_pedidos_previos", "prev_reenc_solicitante",
                   "prev_reenc_rate_solicitante", "n_pedidos_previos_neste_orgao",
                   "prev_reenc_neste_orgao", "n_orgaos_distintos_previos",
                   "dias_desde_ultimo_pedido"]
# Derivadas dentro do featurize a partir do que o chamador envia.
NUM_DERIVED_CALLER = ["prev_reenc_rate_neste_orgao"]

# O QUE O CHAMADOR ENVIA -- diferente da lista de variáveis do modelo.
#
# Fix 7: depois do Fix 1+2 as razões passaram a usar DENOMINADOR MADURO (quantos
# pedidos anteriores já tinham MATURITY_DAYS na data do pedido), não a contagem
# total. O chamador não consegue derivar isso de `n_pedidos_previos`, então
# precisa enviar os denominadores. Sem eles as razões ficavam incoerentes com o
# treinamento -- foi o que produziu a inversão em que o veterano pontuava mais
# que o estreante no exemplo mínimo.
CALLER_INPUTS = [
    # contagens: não dependem de desfecho, não exigem maturação
    "n_pedidos_previos", "n_pedidos_previos_neste_orgao",
    "n_orgaos_distintos_previos", "dias_desde_ultimo_pedido",
    # desfechos maduros e seus denominadores
    "prev_reenc_solicitante", "prev_reenc_solicitante_den",
    "prev_reenc_neste_orgao", "prev_reenc_neste_orgao_den",
]
# Embarcadas no artefato (conduta de entidade pública).
NUM_ORGAO = ["orgao_rate_movel_90d", "orgao_rate_movel_365d",
             "dias_desde_primeiro_pedido_do_orgao"]
NUM_PROD = NUM_BASE + NUM_SOLICITANTE + NUM_DERIVED_CALLER + NUM_ORGAO

# Afetados por H6 -- `Solicitantes` e um RETRATO ATUAL, nao o perfil na abertura:
# zero mudancas em nove campos para 22.963 pessoas ao longo de cinco anos, e
# 100% de registros identicos entre 2022 e 2026. Nao e vazamento de alvo (o
# modelo nao le o rotulo), mas as linhas antigas carregam perfil FUTURO --
# descasamento de tempo de medicao nas covariaveis. `idade` fica fora da lista:
# deriva de DataNascimento, que e invariante por natureza.
CAT_H6 = ["Genero", "Escolaridade", "Profissao", "TipoDemandante",
          "TipoPessoaJuridica", "Pais", "UF_sol", "Municipio_sol"]
NUM_H6 = ["uf_match"]

# CONJUNTO DE PRODUÇÃO -- COM as demográficas, honrando o escopo do TAP.
#
# O Termo de Abertura as inclui explicitamente (§4.1), nomeia escolaridade e
# profissão na justificativa ao cidadão (§4) e constrói o risco central sobre
# elas (§6.1). Removê-las seria desvio de escopo, e a decisão pertence aos
# autores e ao aprovador, não à modelagem.
#
# A mitigação por recência que se cogitou NÃO se sustenta empiricamente
# (scripts/experiment_h6_mitigacao.py): o ganho das demográficas é MAIOR no
# treino de 2022 (+0,0121 de PR-AUC), onde o retrato está ~4,7 anos defasado,
# do que no de 2024 (+0,0052, ~2,7 anos). Sem tendência, logo sem contaminação
# mensurável -- e treinar só em anos recentes custaria desempenho.
#
# H6 segue sendo fato sobre os dados, documentado como limitação em
# docs/VERIFICATION.md, sem efeito detectável no modelo.
CAT_PROD = CAT_BASE
NUM_PROD_FINAL = NUM_PROD

NUM_LEAKY = ["prazo_dias"]
CAT_ASSUNTO = ["AssuntoPedido", "SubAssuntoPedido"]


def load_cohort():
    frames = []
    for y in COHORT:
        ped = _clean(pd.read_csv(_latest(y), usecols=PED_COLS, **READ_KW))
        sol = _clean(pd.read_csv(_latest(y, "SolicitantesPedidos"), usecols=SOL_COLS, **READ_KW))
        sol = sol.rename(columns={"UF": "UF_sol", "Municipio": "Municipio_sol"})
        d = ped.merge(sol.drop_duplicates("IdSolicitante"), on="IdSolicitante", how="left")
        d["ano"] = y
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def fit_organ_rate(train, prior_weight=PRIOR_ORGAO):
    """Taxa histórica suavizada por órgão, ajustada SÓ NO TREINO.

    P1: a implementação mora em `lai_triagem.codificacao`, usada também pelo
    experimento do H10. Aqui fica só o nome antigo, para não quebrar quem
    importa `train.fit_organ_rate`.
    """
    return taxa_suavizada(train, prior_weight)


def encode(df, cats, cat_maps=None):
    """Codifica categóricas em inteiros. Inédito/ausente -> -1, tratado como faltante."""
    out, maps = {}, {}
    for c in cats:
        s = df[c].astype("string")
        m = ({v: i for i, v in enumerate(pd.Index(s.dropna().unique()).sort_values())}
             if cat_maps is None else cat_maps[c])
        out[c] = s.map(m).fillna(-1).astype("int32").to_numpy()
        maps[c] = m
    return out, maps


def evaluate(name, y, p):
    # O ganho tem de ser medido contra a taxa-base DESTA partição: a taxa de
    # positivos cai ao longo do horizonte (8,03% no treino, 5,23% no teste).
    base = float(y.mean())
    print(f"\n  -- {name} --   n={len(y):,}  positivos={int(y.sum()):,}  taxa-base={base*100:.2f}%")
    print(f"     ROC-AUC {roc_auc_score(y, p):.4f}   PR-AUC {average_precision_score(y, p):.4f}")
    rows = []
    for frac in (0.01, 0.02, 0.05, 0.10, 0.20):
        prec, k, n_tied, need = precision_at_k(y, p, frac, return_ties=True)
        # Expõe o tamanho do bloco empatado no corte: se n_tied for grande em
        # relação a `need`, a precisão é uma expectativa, não uma seleção.
        rows.append((f"top {frac*100:.0f}%", k, f"{prec*100:.2f}%", f"{prec/base:.2f}x",
                     n_tied, need))
    print(pd.DataFrame(rows, columns=["fila", "k", "precisao", "ganho",
                                      "empatados_no_corte", "usados_do_empate"]
                       ).to_string(index=False))
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "precision_at": {f"{f:.2f}": precision_at_k(y, p, f)[0] for f in (0.01, 0.05, 0.10)}}


def run_variant(name, cats, nums, tr, va, te, mask):
    print(f"\n{'=' * 78}\nVARIANTE {name}   "
          f"({len(cats)} categóricas + {len(nums)} numéricas)\n{'=' * 78}")
    cols = cats + nums
    e_tr, maps = encode(tr, cats)
    e_va, _ = encode(va, cats, maps)
    e_te, _ = encode(te, cats, maps)

    def mat(d, e):
        m = {c: e[c] for c in cats}
        for c in nums:
            m[c] = pd.to_numeric(d[c], errors="coerce").astype("float32").to_numpy()
        return pd.DataFrame(m, columns=cols)

    Xtr, Xva, Xte = mat(tr, e_tr), mat(va, e_va), mat(te, e_te)
    ytr, yva, yte = tr.y.to_numpy(), va.y.to_numpy(), te.y.to_numpy()
    params = dict(objective="binary", metric="average_precision", learning_rate=0.1,
                  num_leaves=31, max_bin=63, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, min_data_in_leaf=100, num_threads=6, verbose=-1, seed=SEED)
    t0 = time.perf_counter()
    b = lgb.train(params, lgb.Dataset(Xtr, ytr, categorical_feature=cats, free_raw_data=False),
                  num_boost_round=600,
                  valid_sets=[lgb.Dataset(Xva, yva, categorical_feature=cats, free_raw_data=False)],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    secs = time.perf_counter() - t0
    print(f"  ajuste: {secs:.2f}s   melhor iteração={b.best_iteration}   árvores={b.num_trees()}")

    m = {"fit_seconds": round(secs, 2), "best_iteration": b.best_iteration}
    pva, pte = b.predict(Xva), b.predict(Xte)
    m["val"] = evaluate(f"VALIDAÇÃO {VAL_YEAR}", yva, pva)
    m["test"] = evaluate(f"TESTE {TEST_YEAR} (todas as linhas, CENSURADO)", yte, pte)
    m["test_matured"] = evaluate(
        f"TESTE {TEST_YEAR} (maturado: registro <= {MATURITY_DAYS}d antes do retrato)",
        yte[mask], pte[mask])
    print(f"     (censura: {mask.sum():,} de {len(yte):,} maturadas; positivos "
          f"{100*yte[mask].mean():.2f}% maturadas vs {100*yte[~mask].mean():.2f}% recentes)")

    imp = pd.Series(b.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("\n  15 variáveis de maior ganho (%):")
    print((100 * imp / imp.sum()).head(15).round(2).to_string())
    p = ART / f"model_{name}.txt"
    b.save_model(str(p))
    print(f"\n  gravado {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
    return b, maps, m, cols, pte


def fit_calibration(y_val, p_val, y_test, p_test, n_grid=1000):
    """Calibração isotônica ajustada NA VALIDAÇÃO, avaliada no teste maturado.

    Exportada como duas listas de float (grade + imagem), não como objeto
    serializado: mantém o artefato livre de `pickle`, e a aplicação em serviço é
    um `np.interp`.

    NÃO melhora precisão@k. Transformação monótona preserva a ordenação, logo
    preserva a fila. Serve para (a) o número ser legível como probabilidade,
    (b) permitir limiar por custo esperado, (c) viabilizar comparação entre
    modelos distintos, que é pré-requisito de um desenho em dois estágios.
    O efeito sobre precisão@k é medido abaixo para confirmar que é nulo.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val, y_val)
    # Grade de tamanho fixo: o número de limiares da isotônica é ilimitado, e
    # queremos o artefato pequeno e previsível.
    grid = np.linspace(float(p_val.min()), float(p_val.max()), n_grid)
    image = iso.predict(grid)

    def apply(p):
        return np.interp(p, grid, image)

    print(f"\n{'=' * 78}\nCALIBRAÇÃO (isotônica, ajustada na validação {VAL_YEAR})\n{'=' * 78}")
    for label, y, p in (("validação", y_val, p_val), ("teste maturado", y_test, p_test)):
        pc = apply(p)
        print(f"\n  {label}:")
        print(f"    Brier  cru {brier_score_loss(y, p):.5f}  ->  calibrado "
              f"{brier_score_loss(y, pc):.5f}")
        print(f"    média do escore  cru {p.mean():.4f}  calibrado {pc.mean():.4f}"
              f"   (taxa observada {y.mean():.4f})")
        # A fila tem de permanecer a mesma: confirmação empírica da monotonia.
        for frac in (0.01, 0.05, 0.10):
            a, _ = precision_at_k(y, p, frac)
            b, _ = precision_at_k(y, pc, frac)
            print(f"    precisão@{frac*100:.0f}%  cru {a:.4f}  calibrado {b:.4f}"
                  f"   {'idêntica' if abs(a-b) < 1e-9 else f'delta {b-a:+.4f}'}")

    # Confiabilidade por decil no teste, o que a calibração de fato conserta.
    pc_t = apply(p_test)
    dec = pd.qcut(pc_t, 10, labels=False, duplicates="drop")
    rel = pd.DataFrame({"decil": dec, "previsto": pc_t, "observado": y_test}) \
        .groupby("decil").agg(n=("previsto", "size"), previsto=("previsto", "mean"),
                              observado=("observado", "mean")).round(4)
    print("\n  confiabilidade por decil (teste maturado, escore calibrado):")
    print(rel.to_string())
    return {"method": "isotonic", "fitted_on": f"validation_{VAL_YEAR}",
            "grid": [round(float(x), 8) for x in grid],
            "image": [round(float(x), 8) for x in image],
            "brier_raw_test": float(brier_score_loss(y_test, p_test)),
            "brier_calibrated_test": float(brier_score_loss(y_test, pc_t)),
            "note": ("monótona: preserva a ordenação e portanto a precisão@k. "
                     "Serve à legibilidade e ao limiar por custo, não ao ganho.")}


def equity_audit(te, p):
    """A partir do Fix 5, `Escolaridade` NÃO é variável do modelo (H6). Auditar
    por ela ficou mais forte: mede disparidade numa característica que o modelo
    não observa, logo qualquer viés vem da estrutura do problema, não do ajuste."""
    print(f"\n{'=' * 78}\nAUDITORIA DE EQUIDADE — composição da fila de "
          f"{QUEUE_FRAC*100:.0f}% (TAP 6.1)\n{'=' * 78}")
    k = int(round(QUEUE_FRAC * len(te)))
    flagged = te.iloc[np.argsort(-p)[:k]]
    for col in ("Escolaridade", "Genero", "TipoDemandante"):
        pop = te[col].value_counts(normalize=True, dropna=False)
        alert = flagged[col].value_counts(normalize=True, dropna=False)
        cmp = pd.DataFrame({"populacao_%": (100 * pop).round(2),
                            "fila_alerta_%": (100 * alert).round(2)})
        cmp["razao"] = (cmp["fila_alerta_%"] / cmp["populacao_%"]).round(2)
        print(f"\n  {col}:")
        print(cmp.sort_values("populacao_%", ascending=False).head(8).to_string())


def write_metrics_doc(meta, booster, base_metrics):
    """Gera docs/METRICAS.md a partir do artefato.

    Fix 6: a causa dos números obsoletos era eu repetir métricas em prosa em
    quatro documentos. Agora existe UMA fonte gerada, e os outros documentos
    referenciam em vez de repetir. `scripts/check_docs_numbers.py` falha se
    algum texto voltar a contradizer o artefato.
    """
    doc = ROOT / "docs" / "METRICAS.md"
    m = meta["metrics"]["arrival"]
    bm = base_metrics["test_matured"]
    tm = m["test_matured"]

    def pct(x):
        return f"{100*x:.2f}%"

    linhas = [
        "# Métricas — GERADO AUTOMATICAMENTE, NÃO EDITAR À MÃO",
        "",
        f"Gerado por `scripts/train.py` em {meta['created_utc']}.",
        "Qualquer número de desempenho citado em outro documento deve vir daqui.",
        "",
        "## Artefato",
        "",
        "| Item | Valor |",
        "|---|---|",
        f"| Variáveis | **{len(meta['feature_order'])}** |",
        f"| Árvores | **{booster.num_trees()}** |",
        f"| Limiar (fila de {100*meta['queue_fraction']:.0f}%) | **{meta['threshold']}** |",
        f"| Retrato dos dados | `{meta['data_snapshot']}` |",
        f"| Anos de treino | {meta['train_years']} |",
        f"| Maturação | {meta['maturity_days']} dias |",
        f"| Campos excluídos por vazamento | {len(meta['excluded_leakage_features'])} |",
        f"| Campos enviados pelo chamador | {len(meta['caller_supplied_features'])} (atômico) |",
        "",
        f"## Teste {meta['test_year']} maturado — modelo contra linha de base",
        "",
        "| Escore | ROC-AUC | PR-AUC | prec@1% | prec@5% | prec@10% |",
        "|---|---|---|---|---|---|",
        f"| Consulta por órgão (sem modelo) | {bm['roc_auc']:.4f} | {bm['pr_auc']:.4f} | "
        f"{pct(bm['precision_at']['0.01'])} | {pct(bm['precision_at']['0.05'])} | "
        f"{pct(bm['precision_at']['0.10'])} |",
        f"| LightGBM | {tm['roc_auc']:.4f} | {tm['pr_auc']:.4f} | "
        f"{pct(tm['precision_at']['0.01'])} | {pct(tm['precision_at']['0.05'])} | "
        f"{pct(tm['precision_at']['0.10'])} |",
        f"| Diferença relativa | — | {100*(tm['pr_auc']/bm['pr_auc']-1):+.1f}% | "
        f"{100*(tm['precision_at']['0.01']/bm['precision_at']['0.01']-1):+.1f}% | "
        f"**{100*(tm['precision_at']['0.05']/bm['precision_at']['0.05']-1):+.1f}%** | "
        f"{100*(tm['precision_at']['0.10']/bm['precision_at']['0.10']-1):+.1f}% |",
        "",
        "## Custo dos vazamentos (variantes diagnósticas, nunca implantadas)",
        "",
        "| Variante | PR-AUC teste maturado | vs honesta |",
        "|---|---|---|",
    ]
    for nome in ("with_assunto", "LEAKY_with_prazo"):
        v = meta["metrics"].get(nome, {}).get("test_matured")
        if v:
            linhas.append(f"| `{nome}` | {v['pr_auc']:.4f} | "
                          f"{100*(v['pr_auc']-tm['pr_auc']):+.2f} pp |")
    linhas += [
        "",
        "## Ressalvas que não se leem nos números",
        "",
        "- A precisão@k é o **valor esperado** sob desempate uniforme. A linha de",
        "  base tem blocos grandes de empate, porque a taxa por órgão é constante",
        "  dentro do órgão.",
        f"- As variáveis de desfecho usam defasagem de **{meta['maturity_days']} dias**;",
        "  sem ela consumiriam resultados que em produção não seriam conhecidos.",
        "- As variáveis demográficas vêm de um **retrato atual** do cadastro, não do",
        "  perfil na abertura do pedido (H6). Ver `VERIFICATION.md`.",
        "",
    ]
    doc.write_text("\n".join(linhas), encoding="utf-8")
    print(f"gravado docs/METRICAS.md ({doc.stat().st_size/1024:.1f} KB)")


def main():
    t0 = time.perf_counter()
    print("datando a primeira aparição de cada órgão (2012 em diante)...")
    births = organ_birth_table()
    print(f"  órgãos datados: {len(births):,}   mais antigo {min(births.values()).date()}")

    df = build_features(load_cohort(), births)
    print(f"carregado e featurizado em {time.perf_counter() - t0:.1f}s")

    tr = df[df.ano.isin(TRAIN_YEARS)].copy()
    va = df[df.ano.eq(VAL_YEAR)].copy()
    te = df[df.ano.eq(TEST_YEAR)].copy()
    mask = (te._reg <= SNAPSHOT - pd.Timedelta(days=MATURITY_DAYS)).to_numpy()
    print(f"treino {TRAIN_YEARS} n={len(tr):,}  val {VAL_YEAR} n={len(va):,}  "
          f"teste {TEST_YEAR} n={len(te):,} (maturado {mask.sum():,})")
    print(f"taxa de reenc.  treino {tr.y.mean()*100:.2f}%  val {va.y.mean()*100:.2f}%  "
          f"teste {te.y.mean()*100:.2f}%")

    # H10 -- CROSS-FITTING. O mapa do treino inteiro vai para o artefato e para
    # validação/teste, porque é o que a produção consulta. Mas as linhas de
    # TREINO recebem a taxa calculada FORA da própria dobra: sem isso, cada
    # linha carregaria o próprio rótulo na sua maior variável, e o modelo
    # aprenderia a confiar numa `orgao_rate` melhor do que jamais será em
    # produção. Medido em scripts/experiment_h10_crossfit.py: precisão@5% sobe
    # de 21,79% para 24,23%, e a distância para a linha de base encolhe de
    # -3,01 pp para -0,57 pp.
    organ_rate, base = fit_organ_rate(tr)
    for d in (va, te):
        d["orgao_rate"] = d.OrgaoDestinatario.map(organ_rate).fillna(base).astype("float32")
    tr["orgao_rate"] = orgao_rate_crossfit(tr)

    # Tabelas por órgão embarcadas no artefato: estado do órgão no FIM da janela
    # de treino+validação, que é o que um pedido novo deve consultar. Exige
    # reajuste periódico -- ver docs/DECISAO_ESTADO_SOLICITANTE.md.
    recent = pd.concat([tr, va, te]).sort_values("_reg", kind="stable")
    last_per_organ = recent.groupby("OrgaoDestinatario").tail(1).set_index("OrgaoDestinatario")
    organ_movel_90 = last_per_organ["orgao_rate_movel_90d"].astype(float).to_dict()
    organ_movel_365 = last_per_organ["orgao_rate_movel_365d"].astype(float).to_dict()

    print(f"\n{'=' * 78}\nLINHA DE BASE — ordenar por orgao_rate, sem modelo\n{'=' * 78}")
    base_metrics = {
        "val": evaluate(f"VALIDAÇÃO {VAL_YEAR} [base: orgao_rate]", va.y.to_numpy(),
                        va.orgao_rate.to_numpy()),
        "test_matured": evaluate(f"TESTE {TEST_YEAR} maturado [base: orgao_rate]",
                                 te.y.to_numpy()[mask], te.orgao_rate.to_numpy()[mask]),
    }

    b_prod, maps_prod, m_prod, cols_prod, pte = run_variant(
        "arrival", CAT_PROD, NUM_PROD_FINAL, tr, va, te, mask)
    _, _, m_assunto, _, _ = run_variant(
        "with_assunto", CAT_BASE + CAT_ASSUNTO, NUM_PROD, tr, va, te, mask)
    _, _, m_prazo, _, _ = run_variant(
        "LEAKY_with_prazo", CAT_BASE, NUM_PROD + NUM_LEAKY, tr, va, te, mask)
    # Contrafactual de H6: quanto se perderia removendo o que é medido no
    # retrato do cadastro e não na abertura do pedido.
    _, _, m_semdem, _, _ = run_variant(
        "SEM_demografia_H6",
        [c for c in CAT_BASE if c not in CAT_H6],
        [c for c in NUM_PROD if c not in NUM_H6], tr, va, te, mask)

    print(f"\n{'=' * 78}\nCUSTO DOS VAZAMENTOS (variantes diagnósticas)\n{'=' * 78}")
    for label, mm in (("AssuntoPedido", m_assunto), ("prazo_dias", m_prazo)):
        for split in ("val", "test_matured"):
            a, c = m_prod[split], mm[split]
            print(f"  {label:<14} {split:<13} PR-AUC honesta {a['pr_auc']:.4f} vs "
                  f"{c['pr_auc']:.4f}   {100*(c['pr_auc']-a['pr_auc']):+.2f} pp")
    print("  Nenhum dos dois é realizável em produção; ver docs/VERIFICATION.md.")

    print(f"\n{'=' * 78}\nH6 -- CONTRAFACTUAL: O QUE SE PERDERIA "
          f"REMOVENDO AS DEMOGRÁFICAS\n{'=' * 78}")
    for split in ("val", "test_matured"):
        a, c = m_prod[split], m_semdem[split]
        print(f"  {split:<13} PR-AUC produção (com) {a['pr_auc']:.4f} vs sem "
              f"{c['pr_auc']:.4f}   {100*(c['pr_auc']-a['pr_auc']):+.2f} pp")
        print(f"  {'':<13} prec@5% com {100*a['precision_at']['0.05']:.2f}% vs "
              f"sem {100*c['precision_at']['0.05']:.2f}%")
    print("  H6 nao tem efeito mensuravel no desempenho; ver")
    print("  scripts/experiment_h6_mitigacao.py. Producao mantem o escopo do TAP.")

    print(f"\n{'=' * 78}\nMODELO vs LINHA DE BASE (teste maturado)\n{'=' * 78}")
    bm, pm = base_metrics["test_matured"], m_prod["test_matured"]
    print(f"  PR-AUC      base {bm['pr_auc']:.4f}  ->  modelo {pm['pr_auc']:.4f}   "
          f"({100*(pm['pr_auc']/bm['pr_auc']-1):+.1f}%)")
    print(f"  precisão@5% base {bm['precision_at']['0.05']:.4f}  ->  modelo "
          f"{pm['precision_at']['0.05']:.4f}   "
          f"({100*(pm['precision_at']['0.05']/bm['precision_at']['0.05']-1):+.1f}%)")

    equity_audit(te, pte)

    # Limiar do ponto de operação, calibrado na validação.
    pva = b_prod.predict(pd.DataFrame(
        {**{c: encode(va, CAT_PROD, maps_prod)[0][c] for c in CAT_PROD},
         **{c: pd.to_numeric(va[c], errors="coerce").astype("float32").to_numpy()
            for c in NUM_PROD_FINAL}}, columns=cols_prod))
    threshold = float(np.quantile(pva, 1 - QUEUE_FRAC))

    calib = fit_calibration(va.y.to_numpy(), pva, te.y.to_numpy()[mask],
                            b_prod.predict(pd.DataFrame(
                                {**{c: encode(te, CAT_PROD, maps_prod)[0][c] for c in CAT_PROD},
                                 **{c: pd.to_numeric(te[c], errors="coerce")
                                        .astype("float32").to_numpy()
                                    for c in NUM_PROD_FINAL}}, columns=cols_prod))[mask])

    meta = {
        "model_file": "model_arrival.txt",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_snapshot": _latest(TEST_YEAR).name.split("_")[0],
        "train_years": TRAIN_YEARS, "val_year": VAL_YEAR, "test_year": TEST_YEAR,
        "maturity_days": MATURITY_DAYS,
        # Derivadas da variante que REALMENTE treinou o modelo, nunca fixas.
        # Fixá-las produziu artefato declarando 31 variáveis para um modelo de
        # 22, e o serviço quebrava no primeiro predict.
        "feature_order": cols_prod,
        "categorical_features": CAT_PROD,
        "numeric_features": NUM_PROD_FINAL,
        "category_codes": {c: maps_prod[c] for c in CAT_PROD},
        # --- tabelas embarcadas: conduta de entidade pública, sem dado pessoal
        "organ_rate": organ_rate,
        "organ_rate_movel_90d": organ_movel_90,
        "organ_rate_movel_365d": organ_movel_365,
        "organ_birth": {k: v.strftime("%Y-%m-%d") for k, v in births.items()},
        "base_rate": base,
        "threshold": round(threshold, 6),
        "queue_fraction": QUEUE_FRAC,
        # --- dado pessoal: esperado do chamador, nunca embarcado
        "caller_supplied_features": CALLER_INPUTS,
        "caller_supplied_default": -1,
        "caller_supplied_atomic": True,   # Fix 7: conjunto atômico
        "caller_supplied_rationale": "docs/DECISAO_ESTADO_SOLICITANTE.md",
        # Derivadas dentro do featurize a partir de CALLER_INPUTS.
        "derived_from_caller": ["prev_reenc_rate_solicitante",
                                "prev_reenc_rate_neste_orgao"],
        "maturity_days_for_caller_denominators": MATURITY_DAYS,
        "calibration": calib,
        "excluded_leakage_features": [
            "Situacao", "FoiProrrogado", "DataResposta", "Decisao", "EspecificacaoDecisao",
            "DetalhamentoDecisao", "MotivoNegativaAcesso", "PrazoRestricaoAcesso",
            "AssuntoPedido", "SubAssuntoPedido", "Tag", "PrazoAtendimento", "prazo_dias",
            "protocolo_seq",
        ],
        "metrics": {"arrival": m_prod, "with_assunto": m_assunto,
                    "LEAKY_with_prazo": m_prazo, "SEM_demografia_H6": m_semdem,
                    "baseline_orgao_rate": base_metrics},
        "h6_affected_features": CAT_H6 + NUM_H6,
    }
    # Guarda-corpo: artefato incoerente com o modelo e artefato invalido.
    assert len(meta["feature_order"]) == b_prod.num_feature(), (
        f"artefato declara {len(meta['feature_order'])} variaveis, "
        f"modelo treinou com {b_prod.num_feature()}")
    assert set(meta["categorical_features"]) | set(meta["numeric_features"]) \
        == set(meta["feature_order"]), "listas do artefato nao cobrem feature_order"
    (ART / "preprocessor.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    write_metrics_doc(meta, b_prod, base_metrics)
    print(f"\ngravado preprocessor.json ({(ART / 'preprocessor.json').stat().st_size/1024:.0f} KB)")
    print(f"limiar da fila de {QUEUE_FRAC*100:.0f}%: {threshold:.6f}")
    print(f"TEMPO TOTAL: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
