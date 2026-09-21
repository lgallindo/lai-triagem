"""
H10 — `orgao_rate` é codificação de alvo **sem cross-fitting**.

`fit_organ_rate(tr)` calcula a taxa suavizada sobre TODO o treino e depois a
mapeia de volta sobre o próprio treino. Cada linha de treino recebe, portanto,
uma variável que contém o **próprio rótulo** no numerador.

Com `prior_weight=50` a contribuição própria é desprezível num órgão grande e
material num pequeno — e `orgao_rate` é a maior variável do modelo, com ~40%
do ganho. O risco não é vazar rótulo de teste (a codificação sai só do treino);
é o modelo **confiar demais** em `orgao_rate` porque, no treino, ela é melhor
do que jamais será em produção.

Este experimento isola UMA variável: a fórmula de suavização é idêntica nos
dois braços, só muda o cross-fitting.

  atual      taxa do treino inteiro, aplicada ao próprio treino
  crossfit   K-fold: cada dobra recebe a taxa calculada nas OUTRAS dobras.
             Validação e teste seguem recebendo a taxa do treino inteiro, que
             é o que produção faz.

    uv run python scripts/experiment_h10_crossfit.py

Reescreve `artifacts/model_EXP_*.txt`. Restaure com `git checkout -- artifacts/`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train  # noqa: E402

from lai_triagem.config import PRIOR_ORGAO, SEED  # noqa: E402

N_DOBRAS = 5


def taxa_suavizada(quadro: pd.DataFrame, prior: float = PRIOR_ORGAO):
    """Idêntica a `train.fit_organ_rate`, isolada para reuso por dobra."""
    base = quadro.y.mean()
    g = quadro.groupby("OrgaoDestinatario").y.agg(["sum", "count"])
    return ((g["sum"] + prior * base) / (g["count"] + prior)).to_dict(), float(base)


def orgao_rate_crossfit(tr: pd.DataFrame) -> np.ndarray:
    """Taxa por órgão calculada FORA da dobra de cada linha."""
    fora = np.empty(len(tr), dtype="float32")
    kf = KFold(n_splits=N_DOBRAS, shuffle=True, random_state=SEED)
    for treino_idx, teste_idx in kf.split(tr):
        mapa, base = taxa_suavizada(tr.iloc[treino_idx])
        alvo = tr.iloc[teste_idx]
        fora[teste_idx] = alvo.OrgaoDestinatario.map(mapa).fillna(base).to_numpy()
    return fora


print("construindo variáveis...")
df = train.build_features(train.load_cohort(), train.organ_birth_table())
tr = df[df.ano.isin(train.TRAIN_YEARS)].copy()
va = df[df.ano.eq(train.VAL_YEAR)].copy()
te = df[df.ano.eq(train.TEST_YEAR)].copy()
mask = (te._reg <= train.SNAPSHOT - pd.Timedelta(days=train.MATURITY_DAYS)).to_numpy()
print(f"  treino {len(tr):,}  val {len(va):,}  teste {len(te):,} "
      f"(maturado {mask.sum():,})")

# Validação e teste recebem sempre a taxa do treino inteiro: é o que produção
# faz, e não é sobre isso que este experimento decide.
mapa_cheio, base_cheia = taxa_suavizada(tr)
for d in (va, te):
    d["orgao_rate"] = d.OrgaoDestinatario.map(mapa_cheio).fillna(base_cheia).astype("float32")

rate_atual = tr.OrgaoDestinatario.map(mapa_cheio).fillna(base_cheia).astype("float32")
rate_cross = orgao_rate_crossfit(tr)

dif = np.abs(rate_atual.to_numpy() - rate_cross)
print("\ndiferença entre as duas codificações, no treino:")
print(f"  média {dif.mean():.6f}   máximo {dif.max():.6f}")
print(f"  linhas com diferença > 0,01: {int((dif > 0.01).sum()):,} de {len(dif):,}")
print(f"  linhas com diferença > 0,05: {int((dif > 0.05).sum()):,}")

resultados = {}
for nome, valores in (("atual", rate_atual.to_numpy()), ("crossfit", rate_cross)):
    print(f"\n{'#' * 78}\n# VARIANTE {nome}\n{'#' * 78}")
    tr["orgao_rate"] = valores.astype("float32")
    linha_base = train.evaluate(f"[{nome}] LINHA DE BASE orgao_rate",
                                te.y.to_numpy()[mask], te.orgao_rate.to_numpy()[mask])
    _, _, m, _, _ = train.run_variant(f"EXP_H10_{nome}", train.CAT_PROD,
                                      train.NUM_PROD_FINAL, tr, va, te, mask)
    resultados[nome] = {"modelo": m["test_matured"], "base": linha_base}

print(f"\n{'=' * 78}\nRESUMO — teste 2026 maturado\n{'=' * 78}")
cab = f"{'variante':10s} {'PR-AUC':>8s} {'prec@5%':>9s} {'base@5%':>9s} {'modelo-base':>12s}"
print(cab)
print("-" * len(cab))
for nome, r in resultados.items():
    pm = r["modelo"]["pr_auc"]
    p5m = r["modelo"]["precision_at"]["0.05"]
    p5b = r["base"]["precision_at"]["0.05"]
    print(f"{nome:10s} {pm:8.4f} {100*p5m:8.2f}% {100*p5b:8.2f}% "
          f"{100*(p5m - p5b):+11.2f}pp")

print("\nLeitura: se `crossfit` não for pior, vale adotar — a codificação passa")
print("a ser honesta no treino sem custo. Se for MELHOR, o modelo estava sendo")
print("prejudicado por confiar demais numa variável otimista.")
