"""
EXPERIMENTO — modelo em dois estágios por disponibilidade de histórico.

Branch `experimento/dois-estagios`, material para publicação futura. NÃO é o
caminho de produção: `main` segue com estágio único.

MOTIVAÇÃO
45,8% das linhas não têm histórico aproveitável do solicitante (16,9%
anonimizadas, 28,9% de quem pediu uma única vez). No estágio único, essas linhas
recebem o sentinela -1 em oito variáveis e o modelo precisa aprender a tratar
dois regimes com um só conjunto de divisões. Um modelo por regime deveria usar
melhor a capacidade.

A PERGUNTA QUE IMPORTA NÃO É SE CADA MODELO FICA MELHOR
O entregável é UMA fila ranqueada: "os 5% mais arriscados". Dois modelos
treinados em subpopulações diferentes produzem escores em ESCALAS
INCOMPARÁVEIS. Não existe "top 5% conjunto" de dois escores que não estão na
mesma escala. Logo o experimento tem de medir três coisas, não uma:

  E1  cada modelo, no seu próprio regime          -> o estágio duplo ajuda?
  E2  escores crus agrupados numa fila só         -> a fila é coerente?
  E3  escores calibrados por regime, agrupados    -> a calibração conserta?

E3 é o desenho que poderia funcionar, e depende de calibração — que em `main`
mediu-se custar 0,56 pp de precisão@5% por causa dos empates da isotônica. O
experimento existe para descobrir se, no caso de dois estágios, esse custo é
compensado pela coerência que ele viabiliza.

    uv run python experiments/dois_estagios.py
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import (  # noqa: E402
    CAT_BASE, NUM_BASE, NUM_DERIVED_CALLER, NUM_ORGAO, NUM_SOLICITANTE,
    MATURITY_DAYS, SNAPSHOT, TRAIN_YEARS, VAL_YEAR, TEST_YEAR, SEED,
    build_features, encode, fit_organ_rate, load_cohort, organ_birth_table,
    precision_at_k,
)

NUM_FULL = NUM_BASE + NUM_SOLICITANTE + NUM_DERIVED_CALLER + NUM_ORGAO
# No regime sem histórico as variáveis de solicitante são todas -1: constantes,
# logo inúteis. O modelo B usa só o que varia.
NUM_SEM_HIST = NUM_BASE + NUM_ORGAO

PARAMS = dict(objective="binary", metric="average_precision", learning_rate=0.1,
              num_leaves=31, max_bin=63, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, min_data_in_leaf=100, num_threads=6, verbose=-1, seed=SEED)


def fit(tr, va, cats, nums):
    e_tr, maps = encode(tr, cats)
    e_va, _ = encode(va, cats, maps)
    cols = cats + nums

    def mat(d, e):
        m = {c: e[c] for c in cats}
        for c in nums:
            m[c] = pd.to_numeric(d[c], errors="coerce").astype("float32").to_numpy()
        return pd.DataFrame(m, columns=cols)

    b = lgb.train(PARAMS,
                  lgb.Dataset(mat(tr, e_tr), tr.y.to_numpy(), categorical_feature=cats,
                              free_raw_data=False),
                  num_boost_round=600,
                  valid_sets=[lgb.Dataset(mat(va, e_va), va.y.to_numpy(),
                                          categorical_feature=cats, free_raw_data=False)],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    return b, maps, cols, nums, cats


def pred(b, maps, cols, nums, cats, d):
    e, _ = encode(d, cats, maps)
    m = {c: e[c] for c in cats}
    for c in nums:
        m[c] = pd.to_numeric(d[c], errors="coerce").astype("float32").to_numpy()
    return b.predict(pd.DataFrame(m, columns=cols))


def report(name, y, p):
    base = float(y.mean())
    out = {"pr": average_precision_score(y, p), "auc": roc_auc_score(y, p)}
    for f in (0.01, 0.05, 0.10):
        out[f"p{int(f*100)}"] = precision_at_k(y, p, f)[0]
    print(f"  {name:<44} n={len(y):>7,} base={base:.4f}  PR={out['pr']:.4f} "
          f"AUC={out['auc']:.4f}  p@1%={out['p1']:.4f} p@5%={out['p5']:.4f} p@10%={out['p10']:.4f}")
    return out


def main():
    t0 = time.perf_counter()
    df = build_features(load_cohort(), organ_birth_table())
    tr = df[df.ano.isin(TRAIN_YEARS)].copy()
    va = df[df.ano.eq(VAL_YEAR)].copy()
    te = df[df.ano.eq(TEST_YEAR)].copy()
    organ_rate, base = fit_organ_rate(tr)
    for d in (tr, va, te):
        d["orgao_rate"] = d.OrgaoDestinatario.map(organ_rate).fillna(base).astype("float32")
    mask = (te._reg <= SNAPSHOT - pd.Timedelta(days=MATURITY_DAYS)).to_numpy()
    te = te[mask].copy()

    # Regime: histórico aproveitável é ter ao menos um pedido anterior.
    for d in (tr, va, te):
        d["com_hist"] = d.n_pedidos_previos > 0
    print(f"cobertura do regime COM histórico: treino {100*tr.com_hist.mean():.1f}%  "
          f"val {100*va.com_hist.mean():.1f}%  teste {100*te.com_hist.mean():.1f}%")
    print(f"taxa de reenc.  com hist {te[te.com_hist].y.mean():.4f}  "
          f"sem hist {te[~te.com_hist].y.mean():.4f}")

    print(f"\n{'=' * 104}\nE0 — ESTÁGIO ÚNICO (o que está em main)\n{'=' * 104}")
    b1, m1, c1, n1, k1 = fit(tr, va, CAT_BASE, NUM_FULL)
    p_te_single = pred(b1, m1, c1, n1, k1, te)
    r_single = report("estágio único, teste maturado inteiro", te.y.to_numpy(), p_te_single)
    report("  ... restrito ao regime COM histórico",
           te[te.com_hist].y.to_numpy(), p_te_single[te.com_hist.to_numpy()])
    report("  ... restrito ao regime SEM histórico",
           te[~te.com_hist].y.to_numpy(), p_te_single[~te.com_hist.to_numpy()])

    print(f"\n{'=' * 104}\nE1 — UM MODELO POR REGIME, avaliado no próprio regime\n{'=' * 104}")
    trA, vaA, teA = tr[tr.com_hist], va[va.com_hist], te[te.com_hist]
    trB, vaB, teB = tr[~tr.com_hist], va[~va.com_hist], te[~te.com_hist]
    bA, mA, cA, nA, kA = fit(trA, vaA, CAT_BASE, NUM_FULL)
    bB, mB, cB, nB, kB = fit(trB, vaB, CAT_BASE, NUM_SEM_HIST)
    pA_te, pB_te = pred(bA, mA, cA, nA, kA, teA), pred(bB, mB, cB, nB, kB, teB)
    rA = report("modelo A (com histórico, 31 var)", teA.y.to_numpy(), pA_te)
    rB = report("modelo B (sem histórico, 23 var)", teB.y.to_numpy(), pB_te)
    print(f"\n  ganho de A sobre o estágio único no MESMO subconjunto: "
          f"PR {rA['pr'] - report('', teA.y.to_numpy(), p_te_single[te.com_hist.to_numpy()])['pr']:+.4f}")

    print(f"\n{'=' * 104}\nE2 — FILA ÚNICA com escores CRUS agrupados\n{'=' * 104}")
    pooled_raw = np.empty(len(te))
    idx_a = te.com_hist.to_numpy()
    pooled_raw[idx_a], pooled_raw[~idx_a] = pA_te, pB_te
    r_raw = report("dois estágios, escores crus agrupados", te.y.to_numpy(), pooled_raw)
    print(f"\n  faixas de escore por regime (é aqui que a incomparabilidade aparece):")
    for lbl, p in (("A com hist", pA_te), ("B sem hist", pB_te)):
        print(f"    {lbl}: min={p.min():.4f} p50={np.median(p):.4f} p95={np.quantile(p,.95):.4f} max={p.max():.4f}")
    k = int(round(0.05 * len(te)))
    top = np.argsort(-pooled_raw)[:k]
    share_a = idx_a[top].mean()
    print(f"    composição do top 5% agrupado: {100*share_a:.1f}% do regime A "
          f"(regime A é {100*idx_a.mean():.1f}% da população)")

    print(f"\n{'=' * 104}\nE3 — FILA ÚNICA com escores CALIBRADOS por regime\n{'=' * 104}")
    pA_va, pB_va = pred(bA, mA, cA, nA, kA, vaA), pred(bB, mB, cB, nB, kB, vaB)
    isoA = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pA_va, vaA.y.to_numpy())
    isoB = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pB_va, vaB.y.to_numpy())
    pooled_cal = np.empty(len(te))
    pooled_cal[idx_a], pooled_cal[~idx_a] = isoA.predict(pA_te), isoB.predict(pB_te)
    r_cal = report("dois estágios, calibrados e agrupados", te.y.to_numpy(), pooled_cal)
    top_c = np.argsort(-pooled_cal)[:k]
    print(f"    composição do top 5% calibrado: {100*idx_a[top_c].mean():.1f}% do regime A")

    print(f"\n{'=' * 104}\nVEREDITO\n{'=' * 104}")
    rows = [("estágio único (main)", r_single), ("dois estágios, crus", r_raw),
            ("dois estágios, calibrados", r_cal)]
    print(f"  {'desenho':<30} {'PR-AUC':>8} {'p@1%':>8} {'p@5%':>8} {'p@10%':>8}")
    for lbl, r in rows:
        print(f"  {lbl:<30} {r['pr']:>8.4f} {r['p1']:>8.4f} {r['p5']:>8.4f} {r['p10']:>8.4f}")
    best = max(rows, key=lambda t: t[1]["p5"])
    print(f"\n  melhor por precisão@5%: {best[0]}")
    print(f"  delta do melhor sobre o estágio único: "
          f"{best[1]['p5'] - r_single['p5']:+.4f} em precisão@5%")
    print("\n  Custos de produção não medidos aqui e que pesam contra o estágio duplo:")
    print("    - duas etiquetas BentoML, dois ciclos de vida, dois limiares")
    print("    - a tabela de calibração passa a ser parte crítica do caminho de")
    print("      ranqueamento, não mais um enfeite de leitura")
    print("    - dois modelos para reauditar a cada retreinamento")
    print(f"\nTEMPO TOTAL: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
