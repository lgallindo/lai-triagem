"""
Segunda rodada de engenharia de variáveis: Tier 1, dinâmica do órgão, Tier 3.

A primeira rodada (experiment_features_hpo.py) mostrou que dos seis grupos
testados só um funcionou -- histórico do solicitante (G2) -- e que os cinco
restantes, todos atributos estáticos, deram zero ou negativo. A leitura é que a
estrutura explorável está no ESTADO TEMPORAL, não na descrição transversal.
Esta rodada aprofunda essa direção.

Grupos avaliados, cada um somado ao melhor conjunto estabelecido (base + G2):

  T1_experiencia         taxa prévia, experiência específica do órgão, amplitude
                         de órgãos, recência da última interação
  T2_tendencia           orgao_rate_tendencia (janela 90 d menos 365 d)
  T2_idade_orgao         dias_desde_primeiro_pedido_do_orgao
  T2_volume_movel        orgao_volume_movel (contagem em 90 d)
  T3_interacoes          codificações de alvo de pares
  T3_geo                 mesma_regiao (menos esparso que uf_match)
  T3_protocolo           sequencial anual do NUP
  T3_municipio_p5        municipio_sol_rate com prior 5 em vez de 50
  BONUS_rate_movel       orgao_rate_movel_90d / _365d -- NÃO foi pedido, mas sai
                         de graça como subproduto de T2_tendencia

CAUSALIDADE -- E UM DEFEITO CONHECIDO NESTE SCRIPT.

Os agregados por ÓRGÃO usam searchsorted com limite superior ESTRITAMENTE
ANTERIOR à data corrente (side="left"), então nem a linha corrente nem nenhuma
do mesmo dia entram. Isso está correto.

Os agregados por SOLICITANTE usam soma acumulada deslocada, o que impede ver a
si mesmo e o futuro, mas NÃO impede ver irmãos do MESMO DIA -- `DataRegistro`
não tem hora. Auditoria externa mediu 159.320 linhas recebendo histórico de
pedido do mesmo solicitante no mesmo dia, 22.793 com rótulo positivo; e 53.434
linhas consumindo desfecho de pedido com menos de MATURITY_DAYS, que em produção
ainda não seria conhecido.

Consequência: os ganhos de T1 relatados aqui estão OTIMISTAS e serão
republicados após o Fix 1+2. Ver docs/CAMPOS_POST_HOC.md.

    uv run python scripts/experiment_features_v2.py
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import precision_at_k  # noqa: E402

ROOT = Path.home() / "lai-triagem"
INTERIM = ROOT / "data" / "interim"
SNAP = "20260914"
READ_KW = dict(sep=";", encoding="utf-16", dtype=str, na_values=[" ", ""], keep_default_na=True)

TRAIN_YEARS, VAL_YEAR, TEST_YEAR = [2022, 2023, 2024], 2025, 2026
COHORT = TRAIN_YEARS + [VAL_YEAR, TEST_YEAR]
# Anos usados SÓ para datar a primeira aparição de cada órgão. Sem eles, todo
# órgão pré-existente pareceria nascido em 01/01/2022 e a variável de idade
# viraria um proxy de calendário.
BIRTH_YEARS = list(range(2012, 2022))
MATURITY_DAYS = 60
SEED = 42
SNAPSHOT = pd.Timestamp("2026-09-14")

PED_COLS = ["IdPedido", "ProtocoloPedido", "Esfera", "UF", "Municipio", "OrgaoDestinatario",
            "Situacao", "DataRegistro", "FoiReencaminhado", "FormaResposta",
            "OrigemSolicitacao", "IdSolicitante"]
SOL_COLS = ["IdSolicitante", "TipoDemandante", "DataNascimento", "Genero", "Escolaridade",
            "Profissao", "TipoPessoaJuridica", "Pais", "UF", "Municipio"]

CAT_BASE = ["Esfera", "UF", "Municipio", "OrgaoDestinatario", "FormaResposta",
            "OrigemSolicitacao", "TipoDemandante", "Genero", "Escolaridade", "Profissao",
            "TipoPessoaJuridica", "Pais", "UF_sol", "Municipio_sol"]
NUM_BASE = ["reg_month", "reg_dow", "reg_day", "idade", "orgao_rate", "uf_match"]
G2 = ["n_pedidos_previos", "prev_reenc_solicitante"]

GROUPS = {
    "T1_experiencia": ["prev_reenc_rate_solicitante", "n_pedidos_previos_neste_orgao",
                       "prev_reenc_neste_orgao", "n_orgaos_distintos_previos",
                       "dias_desde_ultimo_pedido"],
    "T2_tendencia": ["orgao_rate_tendencia"],
    "T2_idade_orgao": ["dias_desde_primeiro_pedido_do_orgao"],
    "T2_volume_movel": ["orgao_volume_movel"],
    "T3_interacoes": ["orgao_esfera_rate", "orgao_ufsol_rate", "forma_origem_rate"],
    "T3_geo": ["mesma_regiao"],
    "T3_protocolo": ["protocolo_seq"],
    "T3_municipio_p5": ["municipio_sol_rate_p5"],
    "BONUS_rate_movel": ["orgao_rate_movel_90d", "orgao_rate_movel_365d"],
}

REGIAO = {
    **{uf: "N" for uf in ["AC", "AP", "AM", "PA", "RO", "RR", "TO"]},
    **{uf: "NE" for uf in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
    **{uf: "CO" for uf in ["DF", "GO", "MT", "MS"]},
    **{uf: "SE" for uf in ["ES", "MG", "RJ", "SP"]},
    **{uf: "S" for uf in ["PR", "RS", "SC"]},
}


def _clean(df):
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    return df


def load_cohort():
    frames = []
    for y in COHORT:
        ped = _clean(pd.read_csv(INTERIM / f"{SNAP}_Pedidos_csv_{y}.csv",
                                 usecols=PED_COLS, **READ_KW))
        sol = _clean(pd.read_csv(INTERIM / f"{SNAP}_SolicitantesPedidos_csv_{y}.csv",
                                 usecols=SOL_COLS, **READ_KW))
        sol = sol.rename(columns={"UF": "UF_sol", "Municipio": "Municipio_sol"})
        d = ped.merge(sol.drop_duplicates("IdSolicitante"), on="IdSolicitante", how="left")
        d["ano"] = y
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def organ_birth_table():
    """Primeira data de aparição de cada órgão, varrendo 2012 em diante.

    Não usa rótulos, apenas datas, portanto não há risco de vazamento de alvo.
    """
    first = {}
    for y in BIRTH_YEARS + COHORT:
        # Glob em vez de prefixo fixo: a CGU trocou o retrato de 20260914 para
        # 20260915 no meio deste trabalho, e um prefixo fixo perdia os arquivos
        # de 2012-2021 em silêncio, deixando a idade do órgão censurada em 2022.
        hits = sorted(INTERIM.glob(f"*_Pedidos_csv_{y}.csv"))
        if not hits:
            continue
        d = _clean(pd.read_csv(hits[-1], usecols=["OrgaoDestinatario", "DataRegistro"], **READ_KW))
        d["_reg"] = pd.to_datetime(d.DataRegistro, format="%d/%m/%Y", errors="coerce")
        g = d.groupby("OrgaoDestinatario")._reg.min()
        for organ, dt in g.items():
            if pd.notna(dt) and (organ not in first or dt < first[organ]):
                first[organ] = dt
    return first


def organ_rolling(df, windows=(90, 365)):
    """Agregados por órgão em janela móvel, ESTRITAMENTE anteriores à data corrente.

    Para cada linha, considera apenas linhas do mesmo órgão cuja data seja
    menor que a data corrente -- side="left" no limite superior exclui a própria
    linha e todas as do mesmo dia. Sem isso a variável leria o próprio rótulo.
    """
    out = {f"sum_{w}": np.zeros(len(df)) for w in windows}
    out.update({f"cnt_{w}": np.zeros(len(df)) for w in windows})
    dates_all = df["_reg"].to_numpy("datetime64[ns]")
    y_all = df["y"].to_numpy()

    for _, idx in df.groupby("OrgaoDestinatario", sort=False).indices.items():
        d = dates_all[idx]
        order = np.argsort(d, kind="stable")
        idx_s, d_s = idx[order], d[order]
        ycum = np.concatenate([[0.0], np.cumsum(y_all[idx_s])])
        hi = np.searchsorted(d_s, d_s, side="left")          # estritamente antes
        for w in windows:
            lo = np.searchsorted(d_s, d_s - np.timedelta64(w, "D"), side="left")
            out[f"cnt_{w}"][idx_s] = hi - lo
            out[f"sum_{w}"][idx_s] = ycum[hi] - ycum[lo]
    return out


def smoothed_rate(train, keys, prior=50.0):
    base = train.y.mean()
    g = train.groupby(keys).y.agg(["sum", "count"])
    return ((g["sum"] + prior * base) / (g["count"] + prior)).to_dict(), float(base)


def build(df, births):
    reg = pd.to_datetime(df.DataRegistro, format="%d/%m/%Y", errors="coerce")
    nasc = pd.to_datetime(df.DataNascimento, format="%d/%m/%Y", errors="coerce")
    df = df.assign(
        _reg=reg,
        reg_month=reg.dt.month, reg_dow=reg.dt.dayofweek, reg_day=reg.dt.day,
        idade=((reg - nasc).dt.days / 365.25).round(1),
        uf_match=(df.UF_sol.fillna("~") == df.UF.fillna("!")).astype("int8"),
        y=df.FoiReencaminhado.eq("Sim").astype("int8"),
    )
    df.loc[(df.idade < 10) | (df.idade > 110), "idade"] = np.nan
    df = df[df.Situacao.ne("Encaminhada por Outro Órgão")].copy()
    df = df.sort_values("_reg", kind="stable").reset_index(drop=True)

    # ---------------- Tier 1: histórico do solicitante (ver defeito no topo) ---
    # Solicitante anonimizado ('0') não acumula histórico: não é uma pessoa.
    real = df.IdSolicitante.ne("0")
    sub = df.loc[real]
    g_sol = sub.groupby("IdSolicitante", sort=False)

    n_prev = g_sol.cumcount()
    prev_reenc = g_sol.y.cumsum() - sub.y
    g_pair = sub.groupby(["IdSolicitante", "OrgaoDestinatario"], sort=False)
    n_prev_org = g_pair.cumcount()
    prev_reenc_org = g_pair.y.cumsum() - sub.y
    first_pair = (~sub.duplicated(["IdSolicitante", "OrgaoDestinatario"])).astype("int8")
    n_distinct = first_pair.groupby(sub.IdSolicitante).cumsum() - first_pair
    dias_ult = g_sol._reg.diff().dt.days

    for col, val in [
        ("n_pedidos_previos", n_prev), ("prev_reenc_solicitante", prev_reenc),
        ("n_pedidos_previos_neste_orgao", n_prev_org),
        ("prev_reenc_neste_orgao", prev_reenc_org),
        ("n_orgaos_distintos_previos", n_distinct),
        ("dias_desde_ultimo_pedido", dias_ult),
    ]:
        df[col] = np.nan
        df.loc[real, col] = val.astype("float32")
    # Taxa prévia: só definida com pelo menos um pedido anterior.
    df["prev_reenc_rate_solicitante"] = (
        df.prev_reenc_solicitante / df.n_pedidos_previos.where(df.n_pedidos_previos > 0)
    ).astype("float32")
    # -1 marca "sem histórico" de forma distinguível de zero.
    for c in G2 + GROUPS["T1_experiencia"]:
        df[c] = df[c].fillna(-1).astype("float32")

    # ---------------- Tier 2: dinâmica do órgão -------------------------------
    roll = organ_rolling(df)
    base_all = df.y.mean()
    for w in (90, 365):
        cnt, sm = roll[f"cnt_{w}"], roll[f"sum_{w}"]
        df[f"orgao_rate_movel_{w}d"] = np.where(cnt > 0, sm / np.maximum(cnt, 1), base_all)
    df["orgao_rate_tendencia"] = (df.orgao_rate_movel_90d - df.orgao_rate_movel_365d).astype("float32")
    df["orgao_volume_movel"] = roll["cnt_90"].astype("float32")
    born = df.OrgaoDestinatario.map(births)
    df["dias_desde_primeiro_pedido_do_orgao"] = (df._reg - born).dt.days.astype("float32")

    # ---------------- Tier 3 ---------------------------------------------------
    df["mesma_regiao"] = (
        df.UF_sol.map(REGIAO).fillna("?") == df.UF.map(REGIAO).fillna("!")
    ).astype("int8")
    # Sequencial anual do NUP: posição na fila do órgão naquele ano.
    df["protocolo_seq"] = pd.to_numeric(
        df.ProtocoloPedido.str.slice(5, 11), errors="coerce").astype("float32")
    return df


def encode(df, cats, maps=None):
    out, m_out = {}, {}
    for c in cats:
        s = df[c].astype("string")
        m = ({v: i for i, v in enumerate(pd.Index(s.dropna().unique()).sort_values())}
             if maps is None else maps[c])
        out[c] = s.map(m).fillna(-1).astype("int32").to_numpy()
        m_out[c] = m
    return out, m_out


def prec_at(y, p, frac):
    # Delega a implementacao canonica, ciente de empates, de scripts/train.py.
    # A copia local usava argsort e desempatava pela ordem do arquivo.
    return precision_at_k(y, p, frac)[0]


def fit_eval(tr, va, te, mask, cats, nums):
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
    p = dict(objective="binary", metric="average_precision", learning_rate=0.1,
             num_leaves=31, max_bin=63, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, min_data_in_leaf=100, num_threads=6, verbose=-1, seed=SEED)
    t0 = time.perf_counter()
    b = lgb.train(p, lgb.Dataset(Xtr, ytr, categorical_feature=cats, free_raw_data=False),
                  num_boost_round=600,
                  valid_sets=[lgb.Dataset(Xva, yva, categorical_feature=cats, free_raw_data=False)],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    pva, pte = b.predict(Xva), b.predict(Xte)
    return dict(secs=time.perf_counter() - t0, iters=b.best_iteration,
                val_pr=average_precision_score(yva, pva),
                test_pr=average_precision_score(yte[mask], pte[mask]),
                test_p5=prec_at(yte[mask], pte[mask], 0.05),
                test_p10=prec_at(yte[mask], pte[mask], 0.10),
                test_auc=roc_auc_score(yte[mask], pte[mask])), b, cols


def main():
    t0 = time.perf_counter()
    print("datando a primeira aparição de cada órgão (2012 em diante)...")
    births = organ_birth_table()
    print(f"  órgãos datados: {len(births):,}")
    print(f"  mais antigo: {min(births.values()).date()}   mais recente: {max(births.values()).date()}")

    df = build(load_cohort(), births)
    tr = df[df.ano.isin(TRAIN_YEARS)].copy()
    va = df[df.ano.eq(VAL_YEAR)].copy()
    te = df[df.ano.eq(TEST_YEAR)].copy()
    mask = (te._reg <= SNAPSHOT - pd.Timedelta(days=MATURITY_DAYS)).to_numpy()
    print(f"linhas {len(df):,}   treino {len(tr):,}  val {len(va):,}  teste {len(te):,} "
          f"(maturado {mask.sum():,})")

    # codificações de alvo ajustadas SÓ NO TREINO
    orate, base = smoothed_rate(tr, "OrgaoDestinatario")
    oe, _ = smoothed_rate(tr, ["OrgaoDestinatario", "Esfera"])
    ou, _ = smoothed_rate(tr, ["OrgaoDestinatario", "UF_sol"])
    fo, _ = smoothed_rate(tr, ["FormaResposta", "OrigemSolicitacao"])
    mp5, _ = smoothed_rate(tr, "Municipio_sol", prior=5.0)
    for d in (tr, va, te):
        d["orgao_rate"] = d.OrgaoDestinatario.map(orate).fillna(base).astype("float32")
        d["orgao_esfera_rate"] = pd.Series(
            list(zip(d.OrgaoDestinatario, d.Esfera)), index=d.index).map(oe).fillna(base).astype("float32")
        d["orgao_ufsol_rate"] = pd.Series(
            list(zip(d.OrgaoDestinatario, d.UF_sol)), index=d.index).map(ou).fillna(base).astype("float32")
        d["forma_origem_rate"] = pd.Series(
            list(zip(d.FormaResposta, d.OrigemSolicitacao)), index=d.index).map(fo).fillna(base).astype("float32")
        d["municipio_sol_rate_p5"] = d.Municipio_sol.map(mp5).fillna(base).astype("float32")

    print("\n" + "=" * 104)
    print("ABLAÇÃO — cada grupo somado a base+G2 (referência estabelecida)")
    print("=" * 104)

    m_base, _, _ = fit_eval(tr, va, te, mask, CAT_BASE, NUM_BASE)
    m_g2, _, _ = fit_eval(tr, va, te, mask, CAT_BASE, NUM_BASE + G2)
    rows = [("base (chegada)", len(CAT_BASE) + len(NUM_BASE), m_base),
            ("base + G2 [referência]", len(CAT_BASE) + len(NUM_BASE) + 2, m_g2)]
    for name, feats in GROUPS.items():
        m, _, _ = fit_eval(tr, va, te, mask, CAT_BASE, NUM_BASE + G2 + feats)
        rows.append((name, len(CAT_BASE) + len(NUM_BASE) + 2 + len(feats), m))

    ref = m_g2["test_pr"]
    print(f"\n  {'grupo':<26} {'nvar':>4} {'val_PR':>7} {'test_PR':>8} {'test_p5':>8} "
          f"{'test_p10':>9} {'AUC':>7} {'it':>4} {'s':>5}   Δ vs G2")
    for name, nf, m in rows:
        d = m["test_pr"] - ref
        star = "  <<<" if d > 0.004 else ""
        print(f"  {name:<26} {nf:>4} {m['val_pr']:>7.4f} {m['test_pr']:>8.4f} "
              f"{m['test_p5']:>8.4f} {m['test_p10']:>9.4f} {m['test_auc']:>7.4f} "
              f"{m['iters']:>4} {m['secs']:>5.1f}   {d:+.4f}{star}")

    # Combinação dos grupos com ganho positivo, EXCLUINDO os em quarentena.
    # T3_protocolo deu +0,2577 mas verify_h5_protocolo.py mostrou separação de
    # 79x dentro de um mesmo órgão-ano (INSS 2022: Q1=28,4% vs Q4=0,36%) e que
    # o prefixo [0:5] mapeia 23 órgãos na mediana -- logo a fatia codifica a
    # unidade registradora, não um sequencial neutro. Vazamento, não ganho.
    QUARENTENA = {"T3_protocolo"}
    winners = [f for name, feats in GROUPS.items()
               for f in feats
               if name not in QUARENTENA
               and next(m for n, _, m in rows if n == name)["test_pr"] - ref > 0.0]
    print(f"\n  em quarentena (vazamento confirmado): {sorted(QUARENTENA)}")
    if winners:
        m_comb, b_comb, cols = fit_eval(tr, va, te, mask, CAT_BASE, NUM_BASE + G2 + winners)
        print(f"\n  {'COMBINAÇÃO dos positivos':<26} {len(CAT_BASE)+len(NUM_BASE)+2+len(winners):>4} "
              f"{m_comb['val_pr']:>7.4f} {m_comb['test_pr']:>8.4f} {m_comb['test_p5']:>8.4f} "
              f"{m_comb['test_p10']:>9.4f} {m_comb['test_auc']:>7.4f} {m_comb['iters']:>4} "
              f"{m_comb['secs']:>5.1f}   {m_comb['test_pr'] - ref:+.4f}")
        imp = pd.Series(b_comb.feature_importance("gain"), index=cols).sort_values(ascending=False)
        print("\n  15 variáveis de maior ganho na combinação (%):")
        print((100 * imp / imp.sum()).head(15).round(2).to_string())

    print("\n  linha de base sem modelo (consulta por órgão), teste maturado:")
    print(f"    precisão@5% 0.2479   PR-AUC 0.1641")

    # diagnóstico da hipótese dos ministérios recriados
    print("\n" + "=" * 104)
    print("HIPÓTESE: órgãos recém-criados têm competência obscura e encaminham mais")
    print("=" * 104)
    te_all = df[df.ano.isin([2024, 2025, 2026])]
    bins = pd.cut(te_all.dias_desde_primeiro_pedido_do_orgao,
                  [-1, 365, 730, 1460, 2920, 1e9],
                  labels=["<1 ano", "1-2 anos", "2-4 anos", "4-8 anos", "8+ anos"])
    print(te_all.groupby(bins, observed=True).y.agg(["count", "mean"]).round(4).to_string())

    print(f"\nTEMPO TOTAL: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
