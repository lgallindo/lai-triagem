"""
H6, mitigação: treinar em anos recentes reduz o dano do retrato defasado?

`Solicitantes` é um retrato de 2026-09 colado em todos os pedidos. A defasagem
entre o perfil registrado e o perfil real na abertura é de ~4,7 anos para uma
linha de 2022 e ~2,7 anos para uma de 2024. Se a contaminação for real, o ganho
preditivo das variáveis demográficas deve CRESCER conforme o ano de treino se
aproxima do retrato -- porque aí o perfil colado está mais perto do verdadeiro.

Desenho: teste FIXO (2026 maturado) e validação FIXA (2025), variando só o ano de
treino. Para cada configuração, treina duas vezes -- com e sem as demográficas
afetadas por H6 -- e reporta a diferença. A tendência dessa diferença ao longo
dos anos é a evidência.

Previsões opostas:
  contaminação real -> ganho das demográficas cresce de 2022 para 2024
  sem contaminação  -> ganho aproximadamente constante

    uv run python scripts/experiment_h6_mitigacao.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import (  # noqa: E402
    CAT_BASE, CAT_H6, MATURITY_DAYS, NUM_H6, NUM_PROD, SNAPSHOT, VAL_YEAR, TEST_YEAR,
    build_features, encode, fit_organ_rate, load_cohort, organ_birth_table,
    precision_at_k, run_variant,
)

CAT_SEM = [c for c in CAT_BASE if c not in CAT_H6]
NUM_SEM = [c for c in NUM_PROD if c not in NUM_H6]

CONFIGS = [
    ("treino 2022 (retrato ~4,7 anos à frente)", [2022]),
    ("treino 2023 (~3,7 anos)", [2023]),
    ("treino 2024 (~2,7 anos)", [2024]),
    ("treino 2023+2024", [2023, 2024]),
    ("treino 2022+2023+2024 (atual)", [2022, 2023, 2024]),
]


def main():
    t0 = time.perf_counter()
    df = build_features(load_cohort(), organ_birth_table())
    va = df[df.ano.eq(VAL_YEAR)].copy()
    te = df[df.ano.eq(TEST_YEAR)].copy()
    mask = (te._reg <= SNAPSHOT - pd.Timedelta(days=MATURITY_DAYS)).to_numpy()
    print(f"validação {VAL_YEAR} n={len(va):,}   teste {TEST_YEAR} maturado n={mask.sum():,}")

    linhas = []
    for rotulo, anos in CONFIGS:
        tr = df[df.ano.isin(anos)].copy()
        # orgao_rate reajustada para CADA configuração: ajustar no treino é a
        # regra, e o treino muda a cada linha desta tabela.
        orate, base = fit_organ_rate(tr)
        for d in (tr, va, te):
            d["orgao_rate"] = d.OrgaoDestinatario.map(orate).fillna(base).astype("float32")

        _, _, m_com, _, _ = run_variant(f"COM_demo_{'_'.join(map(str, anos))}",
                                        CAT_BASE, NUM_PROD, tr, va, te, mask)
        _, _, m_sem, _, _ = run_variant(f"SEM_demo_{'_'.join(map(str, anos))}",
                                        CAT_SEM, NUM_SEM, tr, va, te, mask)
        c, s = m_com["test_matured"], m_sem["test_matured"]
        linhas.append({
            "config": rotulo, "n_treino": len(tr),
            "PR_com": c["pr_auc"], "PR_sem": s["pr_auc"],
            "d_PR": c["pr_auc"] - s["pr_auc"],
            "p5_com": c["precision_at"]["0.05"], "p5_sem": s["precision_at"]["0.05"],
            "d_p5": c["precision_at"]["0.05"] - s["precision_at"]["0.05"],
        })

    print(f"\n{'=' * 104}")
    print("GANHO DAS DEMOGRÁFICAS POR ANO DE TREINO — teste 2026 maturado, fixo")
    print("=" * 104)
    print(f"  {'configuração':<42} {'n treino':>9} {'PR com':>8} {'PR sem':>8} "
          f"{'Δ PR':>8} {'p5 com':>8} {'p5 sem':>8} {'Δ p5':>8}")
    for r in linhas:
        print(f"  {r['config']:<42} {r['n_treino']:>9,} {r['PR_com']:>8.4f} "
              f"{r['PR_sem']:>8.4f} {r['d_PR']:>+8.4f} {r['p5_com']:>8.4f} "
              f"{r['p5_sem']:>8.4f} {r['d_p5']:>+8.4f}")

    # A tendência é a evidência.
    solo = [r for r in linhas if r["config"].startswith("treino 202")
            and "+" not in r["config"]]
    print(f"\n  tendência do ganho (Δ PR-AUC) nos treinos de ano único:")
    for r in solo:
        print(f"    {r['config'][:28]:<28} {r['d_PR']:+.4f}")
    if len(solo) >= 2:
        delta = solo[-1]["d_PR"] - solo[0]["d_PR"]
        print(f"\n  variação de 2022 para 2024: {delta:+.4f}")
        print("  => " + (
            "CONTAMINAÇÃO CONFIRMADA: as demográficas rendem mais quando o retrato\n"
            "     está mais perto da data do pedido. Treinar em anos recentes mitiga."
            if delta > 0.005 else
            "SEM TENDÊNCIA CLARA: a defasagem do retrato não explica o desempenho\n"
            "     das demográficas. Mitigar por recência não se justifica."))

    print(f"\nTEMPO TOTAL: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
