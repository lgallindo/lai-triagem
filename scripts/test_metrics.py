"""Teste de `precision_at_k`, escrito depois de auditoria externa mostrar que a
versão anterior dependia da ordem das linhas no arquivo.

Sem pytest de propósito: roda com `uv run python scripts/test_metrics.py` e sai
com código 1 se algo falhar. O repositório não tinha teste nenhum; este cobre a
função de que TODA métrica publicada depende.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import precision_at_k  # noqa: E402

falhas = []


def check(nome, obtido, esperado, tol=1e-9):
    ok = abs(obtido - esperado) < tol
    print(f"  {'OK  ' if ok else 'FALHA'} {nome}: obtido {obtido:.6f}, esperado {esperado:.6f}")
    if not ok:
        falhas.append(nome)


print("1. sem empates: comporta-se como a definição ingênua")
y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
# top 30% = k=3 -> escores 0.9, 0.8, 0.7 -> positivos 1,1,0 -> 2/3
check("top30% sem empate", precision_at_k(y, p, 0.3)[0], 2 / 3)

print("\n2. empate no corte: devolve o VALOR ESPERADO, não um sorteio")
y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
p = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
# k=3. Acima do corte: só 0.9 (1 positivo). Bloco empatado em 0.5: 4 itens,
# 1 positivo. Precisa de 2 deles -> 1 + 2*(1/4) = 1.5 -> 1.5/3 = 0.5
check("top30% com empate", precision_at_k(y, p, 0.3)[0], 0.5)

print("\n3. independência da ordem das linhas — o defeito que a auditoria achou")
rng = np.random.default_rng(0)
vals = []
for _ in range(60):
    idx = rng.permutation(len(y))
    vals.append(precision_at_k(y[idx], p[idx], 0.3)[0])
espalhamento = max(vals) - min(vals)
print(f"  {'OK  ' if espalhamento < 1e-12 else 'FALHA'} 60 permutações: "
      f"espalhamento {espalhamento:.2e} (deve ser 0)")
if espalhamento >= 1e-12:
    falhas.append("independencia de ordem")

print("\n4. escore constante (o caso da linha de base por órgão)")
# Com todos os escores iguais, a precisão de qualquer fila é a taxa-base.
y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
p = np.full(10, 0.42)
for frac in (0.1, 0.3, 0.5, 1.0):
    check(f"escore constante, top{frac*100:.0f}%", precision_at_k(y, p, frac)[0], 0.2)

print("\n5. casos de borda")
check("k=1 do topo", precision_at_k(np.array([1, 0]), np.array([0.9, 0.1]), 0.5)[0], 1.0)
check("frac=1 devolve a taxa-base", precision_at_k(y, p, 1.0)[0], 0.2)
_, k = precision_at_k(y, p, 0.0001)
print(f"  {'OK  ' if k == 1 else 'FALHA'} frac minúsculo ainda devolve k=1: k={k}")
if k != 1:
    falhas.append("k minimo")
_, k = precision_at_k(y, p, 5.0)
print(f"  {'OK  ' if k == len(y) else 'FALHA'} frac>1 satura em n: k={k}")
if k != len(y):
    falhas.append("k maximo")

print("\n6. diagnóstico de empates é reportado")
_, k, n_tied, need = precision_at_k(y, p, 0.3, return_ties=True)
ok = (n_tied == 10 and need == 3)
print(f"  {'OK  ' if ok else 'FALHA'} bloco empatado detectado: "
      f"n_tied={n_tied} (esperado 10), need={need} (esperado 3)")
if not ok:
    falhas.append("diagnostico de empates")

print("\n" + "=" * 60)
if falhas:
    print(f"FALHOU: {len(falhas)} verificação(ões) -> {falhas}")
    sys.exit(1)
print("TODAS AS VERIFICAÇÕES PASSARAM")
