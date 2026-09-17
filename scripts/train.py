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
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.home() / "lai-triagem"
INTERIM = ROOT / "data" / "interim"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

READ_KW = dict(sep=";", encoding="utf-16", dtype=str, na_values=[" ", ""], keep_default_na=True)
TRAIN_YEARS, VAL_YEAR, TEST_YEAR = [2022, 2023, 2024], 2025, 2026
COHORT = TRAIN_YEARS + [VAL_YEAR, TEST_YEAR]
# Anos lidos SÓ para datar a primeira aparição de cada órgão. Sem eles, órgão
# pré-existente pareceria nascido em 01/01/2022.
BIRTH_YEARS = list(range(2012, 2022))
# Um pedido precisa de tempo para ser reencaminhado; linhas registradas a menos
# dias do retrato têm rótulo censurado à direita.
MATURITY_DAYS = 60
SNAPSHOT = pd.Timestamp("2026-09-14")
SEED = 42
QUEUE_FRAC = 0.10  # ponto de operação: fila dos 10% mais arriscados
# Prior das taxas em janela móvel. Tem de ser idêntico em
# scripts/refresh_organ_tables.py, senão o reajuste desloca a distribuição da
# variável em relação ao que foi treinado.
PRIOR_MOVEL = 20.0

PED_COLS = ["IdPedido", "Esfera", "UF", "Municipio", "OrgaoDestinatario", "Situacao",
            "DataRegistro", "PrazoAtendimento", "FoiReencaminhado", "FormaResposta",
            "OrigemSolicitacao", "IdSolicitante", "AssuntoPedido", "SubAssuntoPedido"]
SOL_COLS = ["IdSolicitante", "TipoDemandante", "DataNascimento", "Genero", "Escolaridade",
            "Profissao", "TipoPessoaJuridica", "Pais", "UF", "Municipio"]

CAT_BASE = ["Esfera", "UF", "Municipio", "OrgaoDestinatario", "FormaResposta",
            "OrigemSolicitacao", "TipoDemandante", "Genero", "Escolaridade", "Profissao",
            "TipoPessoaJuridica", "Pais", "UF_sol", "Municipio_sol"]
NUM_BASE = ["reg_month", "reg_dow", "reg_day", "idade", "orgao_rate", "uf_match"]
# Fornecidas pelo chamador em produção (dado pessoal, não embarcado).
NUM_SOLICITANTE = ["n_pedidos_previos", "prev_reenc_solicitante",
                   "prev_reenc_rate_solicitante", "n_pedidos_previos_neste_orgao",
                   "prev_reenc_neste_orgao", "n_orgaos_distintos_previos",
                   "dias_desde_ultimo_pedido"]
# Embarcadas no artefato (conduta de entidade pública).
NUM_ORGAO = ["orgao_rate_movel_90d", "orgao_rate_movel_365d",
             "dias_desde_primeiro_pedido_do_orgao"]
NUM_PROD = NUM_BASE + NUM_SOLICITANTE + NUM_ORGAO

NUM_LEAKY = ["prazo_dias"]
CAT_ASSUNTO = ["AssuntoPedido", "SubAssuntoPedido"]


def _clean(df):
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    return df


def _latest(year, kind="Pedidos"):
    """Glob em vez de prefixo fixo: a CGU trocou o retrato de 20260914 para
    20260915 durante o trabalho, e prefixo fixo perdia arquivos em silêncio."""
    hits = sorted(INTERIM.glob(f"*_{kind}_csv_{year}.csv"))
    return hits[-1] if hits else None


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


def organ_birth_table():
    """Primeira data de aparição de cada órgão, de 2012 em diante.
    Só usa datas, nunca rótulos, portanto não há risco de vazamento de alvo."""
    first = {}
    for y in BIRTH_YEARS + COHORT:
        f = _latest(y)
        if f is None:
            continue
        d = _clean(pd.read_csv(f, usecols=["OrgaoDestinatario", "DataRegistro"], **READ_KW))
        d["_reg"] = pd.to_datetime(d.DataRegistro, format="%d/%m/%Y", errors="coerce")
        for organ, dt in d.groupby("OrgaoDestinatario")._reg.min().items():
            if pd.notna(dt) and (organ not in first or dt < first[organ]):
                first[organ] = dt
    return first


def organ_rolling(df, windows=(90, 365)):
    """Agregados por órgão em janela móvel, ESTRITAMENTE anteriores à data corrente.

    side="left" no limite superior exclui a própria linha e todas as do mesmo
    dia. Sem isso a variável leria o próprio rótulo.
    """
    out = {}
    for w in windows:
        out[f"cnt_{w}"] = np.zeros(len(df))
        out[f"sum_{w}"] = np.zeros(len(df))
    dates_all = df["_reg"].to_numpy("datetime64[ns]")
    y_all = df["y"].to_numpy()
    for _, idx in df.groupby("OrgaoDestinatario", sort=False).indices.items():
        order = np.argsort(dates_all[idx], kind="stable")
        idx_s = idx[order]
        d_s = dates_all[idx_s]
        ycum = np.concatenate([[0.0], np.cumsum(y_all[idx_s])])
        hi = np.searchsorted(d_s, d_s, side="left")
        for w in windows:
            lo = np.searchsorted(d_s, d_s - np.timedelta64(w, "D"), side="left")
            out[f"cnt_{w}"][idx_s] = hi - lo
            out[f"sum_{w}"][idx_s] = ycum[hi] - ycum[lo]
    return out


def build_features(df, births):
    reg = pd.to_datetime(df.DataRegistro, format="%d/%m/%Y", errors="coerce")
    prazo = pd.to_datetime(df.PrazoAtendimento, format="%d/%m/%Y", errors="coerce")
    nasc = pd.to_datetime(df.DataNascimento, format="%d/%m/%Y", errors="coerce")
    df = df.assign(
        _reg=reg,
        prazo_dias=(prazo - reg).dt.days,   # só para a variante diagnóstica C
        reg_month=reg.dt.month, reg_dow=reg.dt.dayofweek, reg_day=reg.dt.day,
        idade=((reg - nasc).dt.days / 365.25).round(1),
        uf_match=(df.UF_sol.fillna("~") == df.UF.fillna("!")).astype("int8"),
        y=df.FoiReencaminhado.eq("Sim").astype("int8"),
    )
    # Idades implausíveis são erro de dado, não sinal.
    df.loc[(df.idade < 10) | (df.idade > 110), "idade"] = np.nan
    # Descarta linhas em trânsito: o OrgaoDestinatario delas é o receptor.
    df = df[df.Situacao.ne("Encaminhada por Outro Órgão")].copy()
    df = df.sort_values("_reg", kind="stable").reset_index(drop=True)

    # --- histórico do solicitante, estritamente causal -----------------------
    # Soma acumulada DESLOCADA: a linha corrente nunca vê a si mesma nem o
    # futuro. Solicitante anonimizado ('0') não acumula: não é uma pessoa.
    real = df.IdSolicitante.ne("0")
    sub = df.loc[real]
    g_sol = sub.groupby("IdSolicitante", sort=False)
    g_pair = sub.groupby(["IdSolicitante", "OrgaoDestinatario"], sort=False)
    first_pair = (~sub.duplicated(["IdSolicitante", "OrgaoDestinatario"])).astype("int8")
    vals = {
        "n_pedidos_previos": g_sol.cumcount(),
        "prev_reenc_solicitante": g_sol.y.cumsum() - sub.y,
        "n_pedidos_previos_neste_orgao": g_pair.cumcount(),
        "prev_reenc_neste_orgao": g_pair.y.cumsum() - sub.y,
        "n_orgaos_distintos_previos": first_pair.groupby(sub.IdSolicitante).cumsum() - first_pair,
        "dias_desde_ultimo_pedido": g_sol._reg.diff().dt.days,
    }
    for col, v in vals.items():
        df[col] = np.nan
        df.loc[real, col] = v.astype("float32")
    df["prev_reenc_rate_solicitante"] = (
        df.prev_reenc_solicitante / df.n_pedidos_previos.where(df.n_pedidos_previos > 0)
    ).astype("float32")
    # -1 marca "sem histórico", distinguível de zero, e é o que o chamador
    # deve enviar quando não tiver o dado.
    for c in NUM_SOLICITANTE:
        df[c] = df[c].fillna(-1).astype("float32")

    # --- dinâmica do órgão ---------------------------------------------------
    roll = organ_rolling(df)
    base_all = float(df.y.mean())
    for w in (90, 365):
        cnt, sm = roll[f"cnt_{w}"], roll[f"sum_{w}"]
        # Suavização com prior, igual à de orgao_rate. Sem ela, um órgão com um
        # único pedido na janela produz taxa 0,0 ou 1,0: o ensaio de
        # refresh_organ_tables.py mostrou 16,2% dos órgãos oscilando mais de
        # 5 pp por puro ruído de volume baixo. Prior menor que o de orgao_rate
        # (20 contra 50) porque a janela de 90 d tem menos massa.
        df[f"orgao_rate_movel_{w}d"] = (sm + PRIOR_MOVEL * base_all) / (cnt + PRIOR_MOVEL)
    born = df.OrgaoDestinatario.map(births)
    df["dias_desde_primeiro_pedido_do_orgao"] = (df._reg - born).dt.days.astype("float32")
    return df


def fit_organ_rate(train, prior_weight=50.0):
    """Taxa histórica suavizada de reencaminhamento por órgão, ajustada SÓ NO TREINO."""
    base = train.y.mean()
    g = train.groupby("OrgaoDestinatario").y.agg(["sum", "count"])
    return ((g["sum"] + prior_weight * base) / (g["count"] + prior_weight)).to_dict(), float(base)


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


def precision_at_k(y, p, frac):
    k = max(1, int(round(frac * len(y))))
    return float(y[np.argsort(-p)[:k]].mean()), k


def evaluate(name, y, p):
    # O ganho tem de ser medido contra a taxa-base DESTA partição: a taxa de
    # positivos cai ao longo do horizonte (8,03% no treino, 5,23% no teste).
    base = float(y.mean())
    print(f"\n  -- {name} --   n={len(y):,}  positivos={int(y.sum()):,}  taxa-base={base*100:.2f}%")
    print(f"     ROC-AUC {roc_auc_score(y, p):.4f}   PR-AUC {average_precision_score(y, p):.4f}")
    rows = []
    for frac in (0.01, 0.02, 0.05, 0.10, 0.20):
        prec, k = precision_at_k(y, p, frac)
        rows.append((f"top {frac*100:.0f}%", k, f"{prec*100:.2f}%", f"{prec/base:.2f}x"))
    print(pd.DataFrame(rows, columns=["fila", "k", "precisao", "ganho"]).to_string(index=False))
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "precision_at": {f"{f:.2f}": precision_at_k(y, p, f)[0] for f in (0.01, 0.05, 0.10)}}


def run_variant(name, cats, nums, tr, va, te, mask):
    print(f"\n{'=' * 78}\nVARIANTE {name}   ({len(cats)} categóricas + {len(nums)} numéricas)\n{'=' * 78}")
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


def equity_audit(te, p):
    print(f"\n{'=' * 78}\nAUDITORIA DE EQUIDADE — composição da fila de {QUEUE_FRAC*100:.0f}% (TAP 6.1)\n{'=' * 78}")
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

    organ_rate, base = fit_organ_rate(tr)
    for d in (tr, va, te):
        d["orgao_rate"] = d.OrgaoDestinatario.map(organ_rate).fillna(base).astype("float32")

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
        "arrival", CAT_BASE, NUM_PROD, tr, va, te, mask)
    _, _, m_assunto, _, _ = run_variant(
        "with_assunto", CAT_BASE + CAT_ASSUNTO, NUM_PROD, tr, va, te, mask)
    _, _, m_prazo, _, _ = run_variant(
        "LEAKY_with_prazo", CAT_BASE, NUM_PROD + NUM_LEAKY, tr, va, te, mask)

    print(f"\n{'=' * 78}\nCUSTO DOS VAZAMENTOS (variantes diagnósticas)\n{'=' * 78}")
    for label, mm in (("AssuntoPedido", m_assunto), ("prazo_dias", m_prazo)):
        for split in ("val", "test_matured"):
            a, c = m_prod[split], mm[split]
            print(f"  {label:<14} {split:<13} PR-AUC honesta {a['pr_auc']:.4f} vs "
                  f"{c['pr_auc']:.4f}   {100*(c['pr_auc']-a['pr_auc']):+.2f} pp")
    print("  Nenhum dos dois é realizável em produção; ver docs/VERIFICATION.md.")

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
        {**{c: encode(va, CAT_BASE, maps_prod)[0][c] for c in CAT_BASE},
         **{c: pd.to_numeric(va[c], errors="coerce").astype("float32").to_numpy()
            for c in NUM_PROD}}, columns=cols_prod))
    threshold = float(np.quantile(pva, 1 - QUEUE_FRAC))

    meta = {
        "model_file": "model_arrival.txt",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_snapshot": _latest(TEST_YEAR).name.split("_")[0],
        "train_years": TRAIN_YEARS, "val_year": VAL_YEAR, "test_year": TEST_YEAR,
        "maturity_days": MATURITY_DAYS,
        "feature_order": cols_prod,
        "categorical_features": CAT_BASE,
        "numeric_features": NUM_PROD,
        "category_codes": {c: maps_prod[c] for c in CAT_BASE},
        # --- tabelas embarcadas: conduta de entidade pública, sem dado pessoal
        "organ_rate": organ_rate,
        "organ_rate_movel_90d": organ_movel_90,
        "organ_rate_movel_365d": organ_movel_365,
        "organ_birth": {k: v.strftime("%Y-%m-%d") for k, v in births.items()},
        "base_rate": base,
        "threshold": round(threshold, 6),
        "queue_fraction": QUEUE_FRAC,
        # --- dado pessoal: esperado do chamador, nunca embarcado
        "caller_supplied_features": NUM_SOLICITANTE,
        "caller_supplied_default": -1,
        "caller_supplied_rationale": "docs/DECISAO_ESTADO_SOLICITANTE.md",
        "excluded_leakage_features": [
            "Situacao", "FoiProrrogado", "DataResposta", "Decisao", "EspecificacaoDecisao",
            "DetalhamentoDecisao", "MotivoNegativaAcesso", "PrazoRestricaoAcesso",
            "AssuntoPedido", "SubAssuntoPedido", "Tag", "PrazoAtendimento", "prazo_dias",
            "protocolo_seq",
        ],
        "metrics": {"arrival": m_prod, "with_assunto": m_assunto,
                    "LEAKY_with_prazo": m_prazo, "baseline_orgao_rate": base_metrics},
    }
    (ART / "preprocessor.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"\ngravado preprocessor.json ({(ART / 'preprocessor.json').stat().st_size/1024:.0f} KB)")
    print(f"limiar da fila de {QUEUE_FRAC*100:.0f}%: {threshold:.6f}")
    print(f"TEMPO TOTAL: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
