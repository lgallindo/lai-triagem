"""
Codificação de alvo por órgão (`orgao_rate`), com cross-fitting.

**H10.** Até 21/09/2026 a taxa era calculada sobre todo o treino e aplicada de
volta ao próprio treino: cada linha recebia uma variável que continha o
**próprio rótulo** no numerador. Com prior 50 a contribuição própria é
desprezível num órgão grande e material num pequeno — e `orgao_rate` é a maior
variável do modelo.

Não era vazamento de teste: a codificação sempre saiu só do treino. O dano era
outro, e mensurável: no treino a variável era **melhor do que jamais seria em
produção**, e o modelo aprendeu a confiar nela demais.

Medido em `scripts/experiment_h10_crossfit.py`, com a fórmula de suavização
idêntica nos dois braços:

| | PR-AUC | precisão@5% | contra a linha de base |
|---|---|---|---|
| sem cross-fitting | 0,1789 | 21,79% | −3,01 pp |
| **com cross-fitting** | **0,1901** | **24,23%** | **−0,57 pp** |

A diferença na própria variável é minúscula — média 0,0022, e só 9.923 de
391.425 linhas mudam mais de 0,01. Uma mudança pequena na entrada produzindo
uma grande no resultado é precisamente o que se espera quando o modelo estava
apoiado numa variável otimista.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from lai_triagem.config import PRIOR_ORGAO, SEED

N_DOBRAS = 5


def taxa_suavizada(quadro: pd.DataFrame, prior: float = PRIOR_ORGAO
                   ) -> tuple[dict[str, float], float]:
    """Taxa histórica de reencaminhamento por órgão, suavizada com prior.

    Sem a suavização, um órgão com um único pedido produz taxa 0,0 ou 1,0.
    O prior mistura a taxa do órgão com a taxa-base, na proporção do volume.
    """
    base = quadro.y.mean()
    g = quadro.groupby("OrgaoDestinatario").y.agg(["sum", "count"])
    taxa = (g["sum"] + prior * base) / (g["count"] + prior)
    # `to_dict()` devolve `dict[Hashable, Any]` para o mypy; as chaves são nomes
    # de órgão e os valores, taxas. A conversão explícita documenta isso.
    return {str(k): float(v) for k, v in taxa.to_dict().items()}, float(base)


def orgao_rate_crossfit(treino: pd.DataFrame, prior: float = PRIOR_ORGAO,
                        n_dobras: int = N_DOBRAS, seed: int = SEED) -> np.ndarray:
    """Taxa por órgão calculada FORA da dobra de cada linha (H10).

    Só se aplica às linhas de TREINO. Validação, teste e o artefato continuam
    recebendo a taxa do treino inteiro — que é o que a produção consulta.
    """
    fora = np.empty(len(treino), dtype="float32")
    kf = KFold(n_splits=n_dobras, shuffle=True, random_state=seed)
    for dentro_idx, fora_idx in kf.split(treino):
        mapa, base = taxa_suavizada(treino.iloc[dentro_idx], prior)
        alvo = treino.iloc[fora_idx]
        fora[fora_idx] = alvo.OrgaoDestinatario.map(mapa).fillna(base).to_numpy()
    return fora
