"""
Treina o classificador de risco de reencaminhamento de pedidos LAI.

O desenho decorre das auditorias registradas em docs/VERIFICATION.md:

  * AssuntoPedido / SubAssuntoPedido são SAÍDAS da triagem (59,5% ausentes em
    linhas recém-registradas, 0,0% após respondidas) -> EXCLUÍDOS do modelo de
    produção. A variante B os inclui apenas para medir o tamanho do vazamento.
  * prazo_dias (PrazoAtendimento - DataRegistro) também é vazamento: mediana de
    21 d sem prorrogação contra 31 d com, exatamente os +10 d do art. 11 §2 da
    LAI -> EXCLUÍDO. A variante C o inclui só como diagnóstico.
  * OrgaoDestinatario é o órgão ENDEREÇADO, não o destinatário final -> mantido,
    e é o sinal isolado mais forte disponível.
  * Linhas com Situacao == "Encaminhada por Outro Órgão" estão em trânsito e seu
    OrgaoDestinatario é o receptor -> descartadas do treinamento.

Corte temporal, nunca aleatório: o modelo pontua chegadas futuras.
A métrica primária é precisão@k, não AUC, porque o entregável é uma fila de
prioridade limitada pela capacidade de analistas seniores.

Procedimento completo e reproduzível em docs/TREINAMENTO.md.

Os artefatos são gravados em formatos que carregam direto no BentoML:
  artifacts/model_<variante>.txt     texto nativo do LightGBM (portátil, versionável)
  artifacts/preprocessor.json        códigos de categoria + tabela por órgão + limiar
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
SNAP = "20260914"

READ_KW = dict(sep=";", encoding="utf-16", dtype=str, na_values=[" ", ""], keep_default_na=True)

TRAIN_YEARS, VAL_YEAR, TEST_YEAR = [2022, 2023, 2024], 2025, 2026
# Um pedido precisa de tempo para ser reencaminhado. Linhas registradas a menos
# dias do retrato não tiveram esse tempo, logo seu rótulo sofre censura à direita.
MATURITY_DAYS = 60

PED_COLS = ["IdPedido", "Esfera", "UF", "Municipio", "OrgaoDestinatario", "Situacao",
            "DataRegistro", "PrazoAtendimento", "FoiReencaminhado", "FormaResposta",
            "OrigemSolicitacao", "IdSolicitante", "AssuntoPedido", "SubAssuntoPedido"]
SOL_COLS = ["IdSolicitante", "TipoDemandante", "DataNascimento", "Genero", "Escolaridade",
            "Profissao", "TipoPessoaJuridica", "Pais", "UF", "Municipio"]

CAT_BASE = ["Esfera", "UF", "Municipio", "OrgaoDestinatario", "FormaResposta",
            "OrigemSolicitacao", "TipoDemandante", "Genero", "Escolaridade", "Profissao",
            "TipoPessoaJuridica", "Pais", "UF_sol", "Municipio_sol"]
# prazo_dias está EXCLUÍDO: verify_h3_prazo.py mostrou mediana de 21 d sem
# prorrogação contra 31 d com prorrogação -- exatamente os +10 dias do art. 11
# §2 da LAI. PrazoAtendimento é reescrito quando a prorrogação é concedida, ou
# seja, depois da chegada; logo prazo_dias reimportava FoiProrrogado (já na
# lista de exclusão) pela porta dos fundos. Detinha 40,5% do ganho.
NUM_BASE = ["reg_month", "reg_dow", "reg_day", "idade", "orgao_rate", "uf_match"]
NUM_LEAKY = ["prazo_dias"]
CAT_ASSUNTO = ["AssuntoPedido", "SubAssuntoPedido"]


def _clean(df):
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    return df


def load_year(year: int) -> pd.DataFrame:
    ped = _clean(pd.read_csv(INTERIM / f"{SNAP}_Pedidos_csv_{year}.csv", usecols=PED_COLS, **READ_KW))
    sol = _clean(pd.read_csv(INTERIM / f"{SNAP}_SolicitantesPedidos_csv_{year}.csv",
                             usecols=SOL_COLS, **READ_KW))
    sol = sol.rename(columns={"UF": "UF_sol", "Municipio": "Municipio_sol"})
    sol = sol.drop_duplicates("IdSolicitante")
    df = ped.merge(sol, on="IdSolicitante", how="left")
    df["ano"] = year
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    reg = pd.to_datetime(df.DataRegistro, format="%d/%m/%Y", errors="coerce")
    prazo = pd.to_datetime(df.PrazoAtendimento, format="%d/%m/%Y", errors="coerce")
    nasc = pd.to_datetime(df.DataNascimento, format="%d/%m/%Y", errors="coerce")

    df = df.assign(
        _reg=reg,
        prazo_dias=(prazo - reg).dt.days,
        reg_month=reg.dt.month,
        reg_dow=reg.dt.dayofweek,
        reg_day=reg.dt.day,
        idade=((reg - nasc).dt.days / 365.25).round(1),
        uf_match=(df.UF_sol.fillna("~") == df.UF.fillna("!")).astype("int8"),
        y=df.FoiReencaminhado.eq("Sim").astype("int8"),
    )
    # Idades implausíveis são erro de dado, não sinal.
    df.loc[(df.idade < 10) | (df.idade > 110), "idade"] = np.nan
    return df


def fit_organ_rate(train: pd.DataFrame, prior_weight: float = 50.0):
    """Taxa histórica suavizada de reencaminhamento por órgão, ajustada SÓ NO TREINO."""
    base = train.y.mean()
    g = train.groupby("OrgaoDestinatario").y.agg(["sum", "count"])
    rate = (g["sum"] + prior_weight * base) / (g["count"] + prior_weight)
    return rate.to_dict(), float(base)


def encode(df, cats, cat_maps=None):
    """Codifica categóricas em inteiros. Inédito/ausente -> -1, tratado como faltante."""
    out, maps = {}, {}
    for c in cats:
        s = df[c].astype("string")
        if cat_maps is None:
            levels = pd.Index(s.dropna().unique()).sort_values()
            m = {v: i for i, v in enumerate(levels)}
        else:
            m = cat_maps[c]
        out[c] = s.map(m).fillna(-1).astype("int32").to_numpy()
        maps[c] = m
    return out, maps


def precision_at_k(y, p, frac):
    k = max(1, int(round(frac * len(y))))
    idx = np.argsort(-p)[:k]
    return float(y[idx].mean()), k


def evaluate(name, y, p, base=None):
    # CORREÇÃO DE DEFEITO: o ganho tem de ser medido contra a taxa-base DESTA
    # partição. Passar a taxa-base do treino tornava errado todo ganho do ano de
    # teste (subestimado por 8,03/5,24 = 1,53x), porque a taxa de positivos cai
    # ao longo do horizonte temporal.
    base = float(y.mean())
    print(f"\n  -- {name} --   n={len(y):,}  positives={int(y.sum()):,}  base rate={base*100:.2f}%")
    print(f"     ROC-AUC {roc_auc_score(y, p):.4f}   PR-AUC {average_precision_score(y, p):.4f}")
    rows = []
    for frac in (0.01, 0.02, 0.05, 0.10, 0.20):
        prec, k = precision_at_k(y, p, frac)
        rows.append((f"top {frac*100:.0f}%", k, f"{prec*100:.2f}%", f"{prec/base:.2f}x"))
    print(pd.DataFrame(rows, columns=["queue", "k", "precision", "lift"]).to_string(index=False))
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "precision_at": {f"{f:.2f}": precision_at_k(y, p, f)[0] for f in (0.01, 0.05, 0.10)}}


def run_variant(name, feats_cat, feats_num, tr, va, te, organ_rate, base):
    print(f"\n{'=' * 78}\nVARIANT {name}   ({len(feats_cat)} categorical + {len(feats_num)} numeric)\n{'=' * 78}")
    cols = feats_cat + feats_num

    enc_tr, maps = encode(tr, feats_cat)
    enc_va, _ = encode(va, feats_cat, maps)
    enc_te, _ = encode(te, feats_cat, maps)

    def mat(df, enc):
        d = {c: enc[c] for c in feats_cat}
        for c in feats_num:
            d[c] = pd.to_numeric(df[c], errors="coerce").astype("float32").to_numpy()
        return pd.DataFrame(d, columns=cols)

    Xtr, Xva, Xte = mat(tr, enc_tr), mat(va, enc_va), mat(te, enc_te)
    ytr, yva, yte = tr.y.to_numpy(), va.y.to_numpy(), te.y.to_numpy()

    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=feats_cat, free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, categorical_feature=feats_cat, reference=dtr, free_raw_data=False)

    params = dict(objective="binary", metric="average_precision", learning_rate=0.1,
                  num_leaves=31, max_bin=63, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, min_data_in_leaf=100, num_threads=6, verbose=-1, seed=42)

    t0 = time.perf_counter()
    booster = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(40, verbose=False)])
    fit_s = time.perf_counter() - t0
    print(f"  fit: {fit_s:.2f}s   best_iter={booster.best_iteration}   trees={booster.num_trees()}")

    m = {"fit_seconds": round(fit_s, 2), "best_iteration": booster.best_iteration}
    m["val"] = evaluate(f"VALIDATION {VAL_YEAR}", yva, booster.predict(Xva))
    pte = booster.predict(Xte)
    m["test"] = evaluate(f"TEST {TEST_YEAR} (all rows, RIGHT-CENSORED)", yte, pte)

    # Censura à direita: o arquivo de 2026 é um retrato de 2026-09-14, então um
    # pedido de setembro teve dias -- não meses -- para ser reencaminhado. Seu
    # FoiReencaminhado ainda pode virar "Sim", o que deprime a taxa de positivos
    # observada e subestima a precisão. Reavalia nas linhas maturadas.
    mature = (te["_reg"] <= pd.Timestamp("2026-09-14") - pd.Timedelta(days=MATURITY_DAYS)).to_numpy()
    if mature.sum() > 1000:
        m["test_matured"] = evaluate(
            f"TEST {TEST_YEAR} (matured: registered <= {MATURITY_DAYS}d before snapshot)",
            yte[mature], pte[mature])
        print(f"     (censoring check: {mature.sum():,} of {len(yte):,} rows matured; "
              f"positive rate {100*yte[mature].mean():.2f}% matured vs "
              f"{100*yte[~mature].mean():.2f}% unmatured)")

    imp = pd.Series(booster.feature_importance("gain"), index=cols).sort_values(ascending=False)
    print("\n  top 12 features by gain:")
    print((100 * imp / imp.sum()).head(12).round(2).to_string())

    path = ART / f"model_{name}.txt"
    booster.save_model(str(path))
    print(f"\n  saved {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    return booster, maps, m, cols


def fairness(te, p, base):
    print(f"\n{'=' * 78}\nEQUITY AUDIT — alert composition at top-10% queue (TAP 6.1)\n{'=' * 78}")
    k = int(round(0.10 * len(te)))
    flagged = te.iloc[np.argsort(-p)[:k]]
    for col in ("Escolaridade", "Genero", "TipoDemandante"):
        pop = te[col].value_counts(normalize=True, dropna=False)
        alert = flagged[col].value_counts(normalize=True, dropna=False)
        cmp = pd.DataFrame({"population_%": (100 * pop).round(2),
                            "alert_queue_%": (100 * alert).round(2)})
        cmp["ratio"] = (cmp["alert_queue_%"] / cmp["population_%"]).round(2)
        print(f"\n  {col}:")
        print(cmp.sort_values("population_%", ascending=False).head(8).to_string())


def main():
    t0 = time.perf_counter()
    frames = {y: build_features(load_year(y)) for y in TRAIN_YEARS + [VAL_YEAR, TEST_YEAR]}
    print(f"loaded + featurised in {time.perf_counter() - t0:.1f}s")

    all_df = pd.concat(frames.values(), ignore_index=True)
    # Descarta linhas em trânsito: o OrgaoDestinatario delas é o receptor, não o endereçado.
    n0 = len(all_df)
    all_df = all_df[all_df.Situacao.ne("Encaminhada por Outro Órgão")].copy()
    print(f"rows {n0:,} -> {len(all_df):,} after dropping in-transit "
          f"({n0 - len(all_df):,} with Situacao='Encaminhada por Outro Órgão')")

    tr = all_df[all_df.ano.isin(TRAIN_YEARS)].copy()
    va = all_df[all_df.ano.eq(VAL_YEAR)].copy()
    te = all_df[all_df.ano.eq(TEST_YEAR)].copy()
    print(f"train {TRAIN_YEARS} n={len(tr):,}  val {VAL_YEAR} n={len(va):,}  test {TEST_YEAR} n={len(te):,}")
    print(f"reenc rate  train {tr.y.mean()*100:.2f}%  val {va.y.mean()*100:.2f}%  test {te.y.mean()*100:.2f}%")

    organ_rate, base = fit_organ_rate(tr)
    for d in (tr, va, te):
        d["orgao_rate"] = d.OrgaoDestinatario.map(organ_rate).fillna(base).astype("float32")

    # LINHA DE BASE: ordenar só pela taxa histórica do órgão -- um groupby, sem
    # modelo. 76% do ganho do modelo treinado é identidade do órgão; se o
    # LightGBM não superar essa barra, o entregável honesto é uma tabela de
    # consulta, não um sistema de aprendizado de máquina.
    print(f"\n{'=' * 78}\nBASELINE — rank by orgao_rate only (no model)\n{'=' * 78}")
    baseline_metrics = {}
    for label, d in (("VALIDATION " + str(VAL_YEAR), va), ("TEST " + str(TEST_YEAR), te)):
        baseline_metrics[label] = evaluate(f"{label} [baseline: orgao_rate]",
                                           d.y.to_numpy(), d.orgao_rate.to_numpy())
    mature_mask = (te["_reg"] <= pd.Timestamp("2026-09-14") - pd.Timedelta(days=MATURITY_DAYS)).to_numpy()
    baseline_metrics["test_matured"] = evaluate(
        f"TEST {TEST_YEAR} matured [baseline: orgao_rate]",
        te.y.to_numpy()[mature_mask], te.orgao_rate.to_numpy()[mature_mask])

    booster_a, maps_a, m_a, cols_a = run_variant(
        "arrival", CAT_BASE, NUM_BASE, tr, va, te, organ_rate, base)
    booster_b, maps_b, m_b, cols_b = run_variant(
        "with_assunto", CAT_BASE + CAT_ASSUNTO, NUM_BASE, tr, va, te, organ_rate, base)
    # Só diagnóstico: quantifica o quanto a variável vazada de prazo infla o
    # resultado de vitrine. Nunca implantar esta variante.
    booster_c, maps_c, m_c, cols_c = run_variant(
        "LEAKY_with_prazo", CAT_BASE, NUM_BASE + NUM_LEAKY, tr, va, te, organ_rate, base)

    print(f"\n{'=' * 78}\nLEAKAGE COST OF AssuntoPedido\n{'=' * 78}")
    for split in ("val", "test"):
        a, b = m_a[split], m_b[split]
        print(f"  {split}: PR-AUC {a['pr_auc']:.4f} (arrival) vs {b['pr_auc']:.4f} (with assunto)"
              f"   delta {100*(b['pr_auc']-a['pr_auc']):+.2f} pp")
        print(f"        prec@5% {100*a['precision_at']['0.05']:.2f}% vs "
              f"{100*b['precision_at']['0.05']:.2f}%")
    print("  NOTE: variant B's advantage is not realisable in production — at arrival")
    print("        AssuntoPedido is ~60% missing (see docs/VERIFICATION.md).")

    print(f"\n{'=' * 78}\nLEAKAGE COST OF prazo_dias (diagnostic variant C)\n{'=' * 78}")
    for split in ("val", "test", "test_matured"):
        if split in m_a and split in m_c:
            a, c = m_a[split], m_c[split]
            print(f"  {split:<13}: PR-AUC {a['pr_auc']:.4f} (honest) vs {c['pr_auc']:.4f} (leaky)"
                  f"   inflation {100*(c['pr_auc']-a['pr_auc']):+.2f} pp")
            print(f"  {'':<13}  prec@5% {100*a['precision_at']['0.05']:.2f}% vs "
                  f"{100*c['precision_at']['0.05']:.2f}%")
    print("  prazo_dias = PrazoAtendimento - DataRegistro. Median 21d unprorrogated vs")
    print("  31d prorrogated (LAI art.11 par.2 grants exactly +10d), so the deadline is")
    print("  rewritten post-intake and the feature re-encodes FoiProrrogado.")

    fairness(te, booster_a.predict(
        pd.DataFrame({**{c: encode(te, CAT_BASE, maps_a)[0][c] for c in CAT_BASE},
                      **{c: pd.to_numeric(te[c], errors="coerce").astype("float32").to_numpy()
                         for c in NUM_BASE}}, columns=cols_a)), base)

    # Artefato de serviço: tudo de que o serviço BentoML precisa, sem pickle.
    meta = {
        "model_file": "model_arrival.txt",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_snapshot": SNAP,
        "train_years": TRAIN_YEARS, "val_year": VAL_YEAR, "test_year": TEST_YEAR,
        "feature_order": cols_a,
        "categorical_features": CAT_BASE,
        "numeric_features": NUM_BASE,
        "category_codes": {c: maps_a[c] for c in CAT_BASE},
        "organ_rate": organ_rate,
        "base_rate": base,
        "excluded_leakage_features": [
            "Situacao", "FoiProrrogado", "DataResposta", "Decisao", "EspecificacaoDecisao",
            "DetalhamentoDecisao", "MotivoNegativaAcesso", "PrazoRestricaoAcesso",
            "AssuntoPedido", "SubAssuntoPedido", "Tag",
        ],
        "metrics": {"arrival": m_a, "with_assunto": m_b},
    }
    (ART / "preprocessor.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    sz = (ART / "preprocessor.json").stat().st_size / 1024
    print(f"\nsaved preprocessor.json ({sz:.0f} KB)")
    print(f"TOTAL WALL CLOCK: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
