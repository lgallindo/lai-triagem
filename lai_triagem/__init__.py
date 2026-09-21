"""
Núcleo compartilhado entre o treinamento, o serviço e as guardas.

Existe para que cada coisa tenha **uma** implementação. Antes destes módulos,
`_clean` vivia em seis arquivos e `PRIOR_MOVEL` em dois, com um comentário
pedindo que ficassem iguais — e não ficaram: a correção do H9 alcançou uma das
seis cópias, deixando a ferramenta de produção `refresh_organ_tables.py` capaz
de reintroduzir o defeito no artefato.

  config     caminhos (resolvidos de `__file__`, não de `~`) e constantes
  dados      leitura e limpeza dos CSV da CGU
  metricas   precisão@k, a métrica primária do projeto
  featurize  featurização de chegada e barreira de vazamento, compartilhada
             entre treino e serviço

`__init__.py` existe também por um motivo prático: sem ele `lai_triagem` seria
um pacote de espaço de nomes implícito, e o mypy recusa-se a analisar o mesmo
arquivo sob dois nomes de módulo.
"""
