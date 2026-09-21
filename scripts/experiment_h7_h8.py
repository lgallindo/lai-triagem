"""
Mede o efeito de corrigir H7 (desalinhamento de índice) e H8 (prior não causal).

H7 e H8 foram confirmados por `scripts/verify_h7_alinhamento.py`. Este script
responde a pergunta que importa: **o resultado principal muda?** O projeto
conclui hoje que o modelo não supera a consulta por órgão; se o histórico do
solicitante estava sendo atribuído às linhas erradas em 17,3% dos casos, essa
conclusão pode estar apoiada em variáveis estragadas.

Quatro variantes, treinadas do mesmo jeito que a produção:

    atual        como está hoje (H7 e H8 presentes)
    H7           só o alinhamento corrigido
    H8           só o prior corrigido
    H7+H8        as duas

    uv run python scripts/experiment_h7_h8.py

Reescreve `artifacts/model_EXP_*.txt`. Restaure com `git checkout -- artifacts/`.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train  # noqa: E402


def lagged_correto(sub, keys, lag_days=train.MATURITY_DAYS):
    """Igual a `train.lagged_outcome_sums`, mas guarda o RÓTULO da linha antes
    do `merge_asof`, que reinicia o índice. Sem isso, o que volta é posição e
    quem consome trata como rótulo."""
    daily = (sub.groupby(keys + ["_reg"], as_index=False, observed=True)
                .agg(dy=("y", "sum"), dn=("y", "size"))
                .sort_values("_reg", kind="stable"))
    daily["cum_y"] = daily.groupby(keys, observed=True).dy.cumsum()
    daily["cum_n"] = daily.groupby(keys, observed=True).dn.cumsum()
    left = sub[keys + ["_reg"]].copy()
    left["_cut"] = left["_reg"] - pd.Timedelta(days=lag_days)
    left = left.sort_values("_cut", kind="stable")
    rotulos = left.index.to_numpy()
    m = pd.merge_asof(left, daily[keys + ["_reg", "cum_y", "cum_n"]],
                      left_on="_cut", right_on="_reg", by=keys,
                      direction="backward", suffixes=("", "_d"))
    return (pd.Series(m.cum_y.fillna(0.0).to_numpy(), index=rotulos).reindex(sub.index),
            pd.Series(m.cum_n.fillna(0.0).to_numpy(), index=rotulos).reindex(sub.index))


def corrige_h7(df):
    df = df.copy()
    real = df.IdSolicitante.ne("0")
    sub = df.loc[real]
    for col, keys in (("prev_reenc_solicitante", ["IdSolicitante"]),
                      ("prev_reenc_neste_orgao", ["IdSolicitante", "OrgaoDestinatario"])):
        cy, cn = lagged_correto(sub, keys)
        df[col] = np.nan
        df.loc[real, col] = cy.to_numpy(dtype="float32")
        df[col + "_den"] = np.nan
        df.loc[real, col + "_den"] = cn.to_numpy(dtype="float32")
    df["prev_reenc_rate_solicitante"] = (
        df.prev_reenc_solicitante
        / df.prev_reenc_solicitante_den.where(df.prev_reenc_solicitante_den > 0)
    ).astype("float32")
    df["prev_reenc_rate_neste_orgao"] = (
        df.prev_reenc_neste_orgao
        / df.prev_reenc_neste_orgao_den.where(df.prev_reenc_neste_orgao_den > 0)
    ).astype("float32")
    for c in train.NUM_SOLICITANTE + train.NUM_DERIVED_CALLER:
        df[c] = df[c].fillna(-1).astype("float32")
    return df


def corrige_h8(df):
    df = df.copy()
    base_train = float(df.loc[df.ano.isin(train.TRAIN_YEARS), "y"].mean())
    roll = train.organ_rolling(df)
    for w in (90, 365):
        cnt, sm = roll[f"cnt_{w}"], roll[f"sum_{w}"]
        df[f"orgao_rate_movel_{w}d"] = (
            (sm + train.PRIOR_MOVEL * base_train) / (cnt + train.PRIOR_MOVEL)
        ).astype("float32")
    return df


t0 = time.perf_counter()
print("construindo variáveis uma vez...")
base_df = train.build_features(train.load_cohort(), train.organ_birth_table())
print(f"  {len(base_df):,} linhas em {time.perf_counter() - t0:.1f}s")

VARIANTES = [
    ("atual", lambda d: d),
    ("H7", corrige_h7),
    ("H8", corrige_h8),
    ("H7+H8", lambda d: corrige_h8(corrige_h7(d))),
]

resultados = {}
for nome, transforma in VARIANTES:
    print(f"\n{'#' * 78}\n# VARIANTE {nome}\n{'#' * 78}")
    df = transforma(base_df)
    tr = df[df.ano.isin(train.TRAIN_YEARS)].copy()
    va = df[df.ano.eq(train.VAL_YEAR)].copy()
    te = df[df.ano.eq(train.TEST_YEAR)].copy()
    mask = (te._reg <= train.SNAPSHOT - pd.Timedelta(days=train.MATURITY_DAYS)).to_numpy()

    organ_rate, base = train.fit_organ_rate(tr)
    for d in (tr, va, te):
        d["orgao_rate"] = d.OrgaoDestinatario.map(organ_rate).fillna(base).astype("float32")

    linha_base = train.evaluate(f"[{nome}] LINHA DE BASE orgao_rate",
                                te.y.to_numpy()[mask], te.orgao_rate.to_numpy()[mask])
    _, _, m, _, _ = train.run_variant(f"EXP_{nome.replace('+', '_')}",
                                      train.CAT_PROD, train.NUM_PROD_FINAL,
                                      tr, va, te, mask)
    resultados[nome] = {"modelo": m["test_matured"], "base": linha_base}

print(f"\n{'=' * 78}\nRESUMO — teste 2026 maturado\n{'=' * 78}")
cab = (f"{'variante':10s} {'PR-AUC':>8s} {'base':>8s} "
       f"{'prec@5%':>9s} {'base@5%':>9s} {'modelo-base':>12s}")
print(cab)
print("-" * len(cab))
for nome, r in resultados.items():
    pm, pb = r["modelo"]["pr_auc"], r["base"]["pr_auc"]
    p5m = r["modelo"]["precision_at"]["0.05"]
    p5b = r["base"]["precision_at"]["0.05"]
    print(f"{nome:10s} {pm:8.4f} {pb:8.4f} {100*p5m:8.2f}% {100*p5b:8.2f}% "
          f"{100*(p5m - p5b):+11.2f}pp")

print("\nA pergunta que decide: corrigir H7 faz o modelo passar a superar a")
print("consulta por órgão em precisão@5%? Se a coluna 'modelo-base' seguir")
print("negativa, a conclusão do projeto se mantém, agora sobre variáveis certas.")
