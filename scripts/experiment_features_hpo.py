"""
Experimento: variáveis derivadas candidatas + otimização rápida de hiperparâmetros.

Responde a duas perguntas concretas:

  (A) Existe sinalizador indicando se Escolaridade/Profissao foram preenchidos?
      Não existe campo assim nos dados. Ele tem de ser derivado -- e há DOIS
      mecanismos distintos de ausência, que este script separa:
        1. IdSolicitante == '0'  -> solicitante anonimizado, a junção falha
        2. IdSolicitante válido mas campo em branco -> o cidadão não preencheu
      São coisas diferentes e merecem sinalizadores diferentes.

  (B) Quais outras derivadas valem a pena? Ablação por grupo, medida em
      PR-AUC na validação (2025) e no teste maturado (2026).

E por fim a busca de hiperparâmetros mais barata que faz sentido: busca
aleatória de 16 sorteios. Cada ajuste leva ~2 s, então o total fica em ~35 s.

    uv run python scripts/experiment_features_hpo.py
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
MATURITY_DAYS = 60
SEED = 42

PED_COLS = ["IdPedido", "Esfera", "UF", "Municipio", "OrgaoDestinatario", "Situacao",
            "DataRegistro", "FoiReencaminhado", "FormaResposta", "OrigemSolicitacao",
            "IdSolicitante"]
SOL_COLS = ["IdSolicitante", "TipoDemandante", "DataNascimento", "Genero", "Escolaridade",
            "Profissao", "TipoPessoaJuridica", "Pais", "UF", "Municipio"]

CAT_BASE = ["Esfera", "UF", "Municipio", "OrgaoDestinatario", "FormaResposta",
            "OrigemSolicitacao", "TipoDemandante", "Genero", "Escolaridade", "Profissao",
            "TipoPessoaJuridica", "Pais", "UF_sol", "Municipio_sol"]
NUM_BASE = ["reg_month", "reg_dow", "reg_day", "idade", "orgao_rate", "uf_match"]

# Grupos candidatos, avaliados um a um sobre o conjunto base.
GROUPS = {
    "G1_missingness": ["is_anonymous", "escolaridade_missing", "profissao_missing",
                       "genero_missing", "profile_complete"],
    "G2_historico_solicitante": ["n_pedidos_previos", "prev_reenc_solicitante"],
    "G3_volume_orgao": ["orgao_volume"],
    "G4_target_enc_extra": ["municipio_sol_rate", "profissao_rate", "escolaridade_rate"],
    "G5_calendario": ["reg_week", "reg_doy"],
    "G6_geografia": ["pais_brasil"],
}


def _clean(df):
    df.columns = [c.strip() for c in df.columns]
    # H9: testar `dtype == object` NAO funciona -- READ_KW passa `dtype=str` e o
    # pandas moderno devolve o dtype `str` (PDEP-14). A condicao nunca era
    # verdadeira e a funcao era no-op. Ver scripts/train.py e
    # docs/auditorias/2026-09-18-camadas-2-3.md.
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip()
    return df


def load_all():
    frames = []
    for y in TRAIN_YEARS + [VAL_YEAR, TEST_YEAR]:
        ped = _clean(pd.read_csv(INTERIM / f"{SNAP}_Pedidos_csv_{y}.csv",
                                 usecols=PED_COLS, **READ_KW))
        sol = _clean(pd.read_csv(INTERIM / f"{SNAP}_SolicitantesPedidos_csv_{y}.csv",
                                 usecols=SOL_COLS, **READ_KW))
        sol = sol.rename(columns={"UF": "UF_sol", "Municipio": "Municipio_sol"})
        sol = sol.drop_duplicates("IdSolicitante")
        d = ped.merge(sol, on="IdSolicitante", how="left")
        d["ano"] = y
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def smoothed_rate(train, key, prior=50.0):
    """Codificação de alvo suavizada, ajustada só no treino."""
    base = train.y.mean()
    g = train.groupby(key).y.agg(["sum", "count"])
    return ((g["sum"] + prior * base) / (g["count"] + prior)).to_dict(), float(base)


def build(df):
    reg = pd.to_datetime(df.DataRegistro, format="%d/%m/%Y", errors="coerce")
    nasc = pd.to_datetime(df.DataNascimento, format="%d/%m/%Y", errors="coerce")
    df = df.assign(
        _reg=reg,
        reg_month=reg.dt.month, reg_dow=reg.dt.dayofweek, reg_day=reg.dt.day,
        reg_week=reg.dt.isocalendar().week.astype("float32"),
        reg_doy=reg.dt.dayofyear,
        idade=((reg - nasc).dt.days / 365.25).round(1),
        uf_match=(df.UF_sol.fillna("~") == df.UF.fillna("!")).astype("int8"),
        y=df.FoiReencaminhado.eq("Sim").astype("int8"),
        # --- G1: os dois mecanismos de ausência, separados ---
        is_anonymous=df.IdSolicitante.eq("0").astype("int8"),
        escolaridade_missing=df.Escolaridade.isna().astype("int8"),
        profissao_missing=df.Profissao.isna().astype("int8"),
        genero_missing=df.Genero.isna().astype("int8"),
        pais_brasil=df.Pais.eq("Brasil").astype("int8"),
    )
    df["profile_complete"] = (
        (~df.Escolaridade.isna()) & (~df.Profissao.isna()) & (~df.Genero.isna())
    ).astype("int8")
    df.loc[(df.idade < 10) | (df.idade > 110), "idade"] = np.nan

    # --- G2: histórico do solicitante ---
    # ATENÇÃO -- DEFEITO CONHECIDO, correção planejada (Fix 1+2).
    # A soma acumulada é deslocada, então a linha não vê a si mesma nem o
    # futuro. MAS `DataRegistro` é somente data, sem hora, e cumcount/cumsum
    # INCLUEM os irmãos do MESMO DIA, cujo desfecho não seria conhecido em
    # produção. Auditoria externa mediu 159.320 linhas afetadas, 22.793 com
    # rótulo positivo vindo do empate. Os números de G2 abaixo estão, portanto,
    # otimistas. Solicitante anonimizado ('0') não acumula: não é uma pessoa.
    df = df.sort_values("_reg", kind="stable").reset_index(drop=True)
    real = df.IdSolicitante.ne("0")
    g = df[real].groupby("IdSolicitante", sort=False)
    df.loc[real, "n_pedidos_previos"] = g.cumcount().astype("float32")
    df.loc[real, "prev_reenc_solicitante"] = (
        g.y.cumsum() - df.loc[real, "y"]
    ).astype("float32")
    df[["n_pedidos_previos", "prev_reenc_solicitante"]] = \
        df[["n_pedidos_previos", "prev_reenc_solicitante"]].fillna(-1)
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
    return precision_at_k(y, p, frac)[0]


def fit_eval(tr, va, te, te_mask, cats, nums, params=None, rounds=600):
    cols = cats + nums
    enc_tr, maps = encode(tr, cats)
    enc_va, _ = encode(va, cats, maps)
    enc_te, _ = encode(te, cats, maps)

    def mat(df, enc):
        d = {c: enc[c] for c in cats}
        for c in nums:
            d[c] = pd.to_numeric(df[c], errors="coerce").astype("float32").to_numpy()
        return pd.DataFrame(d, columns=cols)

    Xtr, Xva, Xte = mat(tr, enc_tr), mat(va, enc_va), mat(te, enc_te)
    ytr, yva, yte = tr.y.to_numpy(), va.y.to_numpy(), te.y.to_numpy()

    p = params or dict(objective="binary", metric="average_precision", learning_rate=0.1,
                       num_leaves=31, max_bin=63, feature_fraction=0.8,
                       bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=100,
                       num_threads=6, verbose=-1, seed=SEED)
    t0 = time.perf_counter()
    b = lgb.train(p, lgb.Dataset(Xtr, ytr, categorical_feature=cats, free_raw_data=False),
                  num_boost_round=rounds,
                  valid_sets=[lgb.Dataset(Xva, yva, categorical_feature=cats, free_raw_data=False)],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    secs = time.perf_counter() - t0
    pva, pte = b.predict(Xva), b.predict(Xte)
    return dict(
        secs=secs, iters=b.best_iteration,
        val_pr=average_precision_score(yva, pva),
        val_p5=prec_at(yva, pva, 0.05),
        test_pr=average_precision_score(yte[te_mask], pte[te_mask]),
        test_p5=prec_at(yte[te_mask], pte[te_mask], 0.05),
        test_auc=roc_auc_score(yte[te_mask], pte[te_mask]),
    ), b, cols


def main():
    t_all = time.perf_counter()
    df = build(load_all())
    df = df[df.Situacao.ne("Encaminhada por Outro Órgão")].copy()

    tr = df[df.ano.isin(TRAIN_YEARS)].copy()
    va = df[df.ano.eq(VAL_YEAR)].copy()
    te = df[df.ano.eq(TEST_YEAR)].copy()
    te_mask = (te._reg <= pd.Timestamp("2026-09-14") - pd.Timedelta(days=MATURITY_DAYS)).to_numpy()

    # ---------------------------------------------------------------- (A)
    print("=" * 78)
    print("(A) MECANISMOS DE AUSÊNCIA — existe sinalizador nativo? NÃO. Derivados:")
    print("=" * 78)
    n = len(df)
    anon = df.is_anonymous.eq(1)
    print(f"  linhas                                        {n:,}")
    print(f"  IdSolicitante == '0' (anonimizado)            "
          f"{anon.sum():,} ({100*anon.mean():.2f}%)")
    print(f"  Escolaridade ausente                          {df.escolaridade_missing.sum():,}"
          f" ({100*df.escolaridade_missing.mean():.2f}%)")
    print(f"    ...destes, com IdSolicitante VÁLIDO         "
          f"{(df.escolaridade_missing.eq(1) & ~anon).sum():,}"
          f" ({100*(df.escolaridade_missing.eq(1) & ~anon).mean():.2f}%)")
    print("    -> ou seja: a junção funcionou e o cidadão simplesmente não preencheu.")
    print(f"  perfil completo (escol.+prof.+gênero)         {df.profile_complete.sum():,}"
          f" ({100*df.profile_complete.mean():.2f}%)")
    print("\n  taxa de reencaminhamento por mecanismo:")
    print(df.groupby(["is_anonymous", "escolaridade_missing"]).y
          .agg(["count", "mean"]).round(4).to_string())

    # taxa por número de pedidos anteriores (o histórico é informativo?)
    print("\n  taxa de reencaminhamento por nº de pedidos anteriores do solicitante:")
    bins = pd.cut(df.n_pedidos_previos, [-2, -0.5, 0.5, 2.5, 10.5, 50.5, 1e9],
                  labels=["anônimo", "0", "1-2", "3-10", "11-50", "50+"])
    print(df.groupby(bins, observed=True).y.agg(["count", "mean"]).round(4).to_string())

    # ---------------------------------------------------------------- (B)
    print("\n" + "=" * 78)
    print("(B) ABLAÇÃO DE VARIÁVEIS DERIVADAS (base = conjunto de chegada atual)")
    print("=" * 78)

    orate, base = smoothed_rate(tr, "OrgaoDestinatario")
    mrate, _ = smoothed_rate(tr, "Municipio_sol")
    prate, _ = smoothed_rate(tr, "Profissao")
    erate, _ = smoothed_rate(tr, "Escolaridade")
    vol = np.log1p(tr.groupby("OrgaoDestinatario").size()).to_dict()
    for d in (tr, va, te):
        d["orgao_rate"] = d.OrgaoDestinatario.map(orate).fillna(base).astype("float32")
        d["municipio_sol_rate"] = d.Municipio_sol.map(mrate).fillna(base).astype("float32")
        d["profissao_rate"] = d.Profissao.map(prate).fillna(base).astype("float32")
        d["escolaridade_rate"] = d.Escolaridade.map(erate).fillna(base).astype("float32")
        d["orgao_volume"] = d.OrgaoDestinatario.map(vol).fillna(0.0).astype("float32")

    rows = []
    m0, _, _ = fit_eval(tr, va, te, te_mask, CAT_BASE, NUM_BASE)
    rows.append(("base (20 var)", len(CAT_BASE) + len(NUM_BASE), m0))
    for name, feats in GROUPS.items():
        m, _, _ = fit_eval(tr, va, te, te_mask, CAT_BASE, NUM_BASE + feats)
        rows.append((name, len(CAT_BASE) + len(NUM_BASE) + len(feats), m))
    # tudo junto
    allf = [f for fs in GROUPS.values() for f in fs]
    m_all, _, _ = fit_eval(tr, va, te, te_mask, CAT_BASE, NUM_BASE + allf)
    rows.append(("TODOS os grupos", len(CAT_BASE) + len(NUM_BASE) + len(allf), m_all))

    print(f"\n  {'variante':<26} {'nvar':>5} {'val_PR':>8} {'test_PR':>8} {'test_p5':>8} "
          f"{'test_AUC':>9} {'iters':>6} {'s':>6}   Δtest_PR")
    for name, nf, m in rows:
        d = m["test_pr"] - m0["test_pr"]
        print(f"  {name:<26} {nf:>5} {m['val_pr']:>8.4f} {m['test_pr']:>8.4f} "
              f"{m['test_p5']:>8.4f} {m['test_auc']:>9.4f} {m['iters']:>6} {m['secs']:>6.1f}"
              f"   {d:+.4f}")

    # ---------------------------------------------------------------- (C)
    print("\n" + "=" * 78)
    print("(C) BUSCA ALEATÓRIA DE 16 SORTEIOS — a otimização mais barata que presta")
    print("=" * 78)
    rng = np.random.default_rng(SEED)
    space = dict(
        num_leaves=[15, 31, 63, 127],
        learning_rate=[0.03, 0.05, 0.1, 0.2],
        min_data_in_leaf=[20, 50, 100, 300],
        feature_fraction=[0.6, 0.8, 1.0],
        max_bin=[31, 63, 127],
    )
    best = (m0["val_pr"], "base", None)
    trials = []
    for i in range(16):
        p = dict(objective="binary", metric="average_precision", bagging_fraction=0.8,
                 bagging_freq=1, num_threads=6, verbose=-1, seed=SEED)
        for k, v in space.items():
            p[k] = v[rng.integers(len(v))]
        m, _, _ = fit_eval(tr, va, te, te_mask, CAT_BASE, NUM_BASE, params=p)
        trials.append((m["val_pr"], m["test_pr"], m["test_p5"], m["secs"], p))
        if m["val_pr"] > best[0]:
            best = (m["val_pr"], f"sorteio {i}", p)
    trials.sort(key=lambda t: -t[0])
    print(f"\n  {'val_PR':>8} {'test_PR':>8} {'test_p5':>8} {'s':>5}  hiperparâmetros")
    for vp, tp, t5, s, p in trials[:6]:
        sh = {k: p[k] for k in ("num_leaves", "learning_rate", "min_data_in_leaf",
                                "feature_fraction", "max_bin")}
        print(f"  {vp:>8.4f} {tp:>8.4f} {t5:>8.4f} {s:>5.1f}  {sh}")
    print(f"\n  linha de partida (val_PR)   {m0['val_pr']:.4f}")
    print(f"  melhor da busca (val_PR)    {best[0]:.4f}   ({best[1]})")
    print(f"  ganho em val_PR             {best[0] - m0['val_pr']:+.4f}")
    print(f"\nTEMPO TOTAL DO EXPERIMENTO: {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()
