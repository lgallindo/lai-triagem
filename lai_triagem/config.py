"""
Caminhos e constantes, num lugar só.

**P5 — por que não `Path.home()`.** Onze arquivos resolviam o repositório como
`Path.home() / "lai-triagem"`, o que só funciona se ele estiver exatamente ali.
Quem clonar em qualquer outro lugar — e o auditor externo apontou isso — vê o
código procurar dados numa pasta que não existe. Aqui `RAIZ` sai de
`__file__`, então o repositório funciona onde estiver.

**P1 — por que as constantes vivem aqui.** `PRIOR_MOVEL` estava declarado em
dois arquivos, com um comentário dizendo "tem de ser idêntico ao do outro".
Invariante garantida por comentário é invariante que um dia diverge — e neste
projeto divergiu: o `_clean` estava em seis cópias e a correção do H9 alcançou
uma só, deixando a ferramenta de produção `refresh_organ_tables.py` capaz de
reintroduzir o defeito no artefato. Constante compartilhada mora num módulo,
não num comentário.
"""

from pathlib import Path

import pandas as pd

# `config.py` está em `lai_triagem/`, logo a raiz é o diretório-pai dele.
RAIZ = Path(__file__).resolve().parents[1]
INTERIM = RAIZ / "data" / "interim"
ART = RAIZ / "artifacts"

# Os CSV da CGU são UTF-16 com `;`. `dtype=str` mantém tudo como texto: a
# conversão é feita depois, com controle. Ver `dados.limpar` sobre o efeito
# desse `dtype=str` no corte de espaços (H9).
READ_KW = dict(sep=";", encoding="utf-16", dtype=str,
               na_values=[" ", ""], keep_default_na=True)

# --------------------------------------------------------- corte temporal
TRAIN_YEARS = [2022, 2023, 2024]
VAL_YEAR = 2025
TEST_YEAR = 2026
COHORT = [*TRAIN_YEARS, VAL_YEAR, TEST_YEAR]
# Anos lidos SÓ para datar a primeira aparição de cada órgão. Sem eles, órgão
# pré-existente pareceria nascido em 01/01/2022.
BIRTH_YEARS = list(range(2012, 2022))

# ------------------------------------------------------------- maturação
# Um pedido precisa de tempo para ser reencaminhado; linhas registradas a menos
# dias do retrato têm rótulo censurado à direita.
MATURITY_DAYS = 60
SNAPSHOT = pd.Timestamp("2026-09-14")

# ------------------------------------------------------------- modelagem
SEED = 42
QUEUE_FRAC = 0.10  # ponto de operação: fila dos 10% mais arriscados
# Prior das taxas em janela móvel. Usado pelo treinamento E pelo reajuste de
# tabelas; por isso está aqui, e não duplicado nos dois.
PRIOR_MOVEL = 20.0
# Prior da codificação por órgão (`orgao_rate`).
PRIOR_ORGAO = 50.0
