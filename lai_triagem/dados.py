"""
Leitura e limpeza dos CSV da CGU — uma implementação, não seis.

Esta função existia em **seis** arquivos. A correção do H9 em 18/09/2026
alcançou apenas `scripts/train.py`; as outras cinco ficaram com a versão que
não limpava nada. A pior delas era `scripts/refresh_organ_tables.py`, que **não
é experimento**: é a ferramenta de produção que reescreve as tabelas de órgão
dentro do artefato. Rodá-la teria devolvido as chaves com espaço e quebrado de
novo, em silêncio, os 41 órgãos que o serviço passou a encontrar.

A correção do H9 tinha prazo de validade de três dias, e ninguém saberia.
"""

from pathlib import Path

import pandas as pd

from lai_triagem.config import INTERIM, READ_KW


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    """Corta espaço das bordas de todo texto, e do nome das colunas.

    **H9.** A versão anterior testava `df[c].dtype == object` antes de cortar.
    `READ_KW` passa `dtype=str`, e o pandas moderno devolve o dtype `str`
    (PDEP-14), **não** `object`: a condição nunca era verdadeira e a função era
    no-op. Parecia proteção, e não fazia nada.

    O custo: 162 chaves com espaço nas bordas ficaram nas tabelas do artefato.
    O serviço corta o texto que recebe, então 41 órgãos nunca eram encontrados
    e recaíam na taxa-base — 1,41% dos pedidos de 2026. Pior, três identidades
    ficaram partidas em duas no próprio treino: `Prefeitura Municipal` aparecia
    como 1.878 pedidos com espaço e 16.446 sem, como se fossem órgãos
    diferentes.

    `select_dtypes` em vez de comparar dtype à mão, para não depender de qual
    representação de texto o pandas resolva usar na próxima versão.
    """
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip()
    return df


def arquivo_mais_recente(ano: int, tipo: str = "Pedidos") -> Path | None:
    """O arquivo daquele ano, seja qual for o prefixo de retrato.

    Glob em vez de prefixo fixo: a CGU trocou o retrato de `20260914` para
    `20260915` no meio do trabalho, e um prefixo fixo perdia arquivos em
    silêncio — a idade dos órgãos ficou censurada em 2022 sem aviso.
    """
    achados = sorted(INTERIM.glob(f"*_{tipo}_csv_{ano}.csv"))
    return achados[-1] if achados else None


def ler(ano: int, colunas: list[str], tipo: str = "Pedidos") -> pd.DataFrame:
    """Lê e limpa de uma vez, que é como todo chamador usa."""
    return limpar(pd.read_csv(arquivo_mais_recente(ano, tipo),
                              usecols=colunas, **READ_KW))
