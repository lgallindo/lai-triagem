"""
Métrica primária do projeto: precisão na fila dos k primeiros.

Mora no pacote, e não em `scripts/train.py`, porque `test_metrics.py` e
`experiment_features_hpo.py` a importavam de lá com `from scripts.train import
...` — o que arrasta o módulo de treinamento inteiro, com seus efeitos de
import, só para usar uma função de vinte linhas.
"""

import numpy as np


def precision_at_k(y, p, frac, return_ties=False):
    """Precisão na fila dos k primeiros, com desempate EXPLÍCITO.

    Defeito corrigido (apontado por auditoria externa): a versão anterior usava
    `np.argsort(-p)[:k]`, que desempata pela ordem das linhas no arquivo. O
    escore da linha de base é a taxa por órgão, CONSTANTE dentro de cada órgão,
    logo há blocos enormes de empate exatamente no ponto de corte -- e o número
    passava a depender da ordem de leitura do CSV, não do modelo.

    Aqui devolvemos o VALOR ESPERADO sob desempate uniforme: todos os positivos
    com escore estritamente acima do corte, mais a fração proporcional do bloco
    empatado. É determinístico e bem definido, e não depende de ordenação.
    """
    n = len(y)
    k = min(max(1, int(round(frac * n))), n)
    order = np.argsort(-p, kind="stable")
    ps, ys = np.asarray(p)[order], np.asarray(y)[order]

    cut = ps[k - 1]
    above = ps > cut
    n_above = int(above.sum())
    pos_above = float(ys[above].sum())

    tied = ps == cut
    n_tied = int(tied.sum())
    pos_tied = float(ys[tied].sum())

    need = k - n_above                      # quantos do bloco empatado entram
    exp_pos = pos_above + (need * pos_tied / n_tied if n_tied else 0.0)
    prec = exp_pos / k
    if return_ties:
        # n_tied >> need indica que o corte cai no meio de um bloco grande: o
        # número é uma expectativa, não uma seleção.
        return prec, k, n_tied, need
    return prec, k
