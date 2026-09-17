# Auditoria de vazamento e viabilidade — risco de reencaminhamento LAI

Toda afirmação aqui é reproduzível a partir de `scripts/` contra o retrato
**20260914** dos dados abertos do Fala.BR. Fonte:
<https://dadosabertos-download.cgu.gov.br/FalaBR/Arquivos_FalaBR/>

Coorte: `Pedidos_csv_{2022..2026}`, **655.177 linhas**, taxa-base de
reencaminhamento **7,32%**. As 459 linhas com
`Situacao == "Encaminhada por Outro Órgão"` são descartadas da modelagem: estão
em trânsito, logo seu `OrgaoDestinatario` é o receptor, não o endereçado.

## Resumo dos achados

| ID | Hipótese | Veredito | Script |
|----|-----------|---------|--------|
| H1 | `AssuntoPedido` é atribuído durante a triagem, não na chegada | **CONFIRMADA — excluído** | `verify_leakage_v2.py` |
| H2 | `OrgaoDestinatario` é sobrescrito no reencaminhamento | **REFUTADA — mantido** | `verify_h2_final.py`, `verify_h2_baserate.py` |
| H3 | `prazo_dias` é variável de chegada | **REFUTADA — é vazamento, excluído** | `verify_h3_prazo.py` |
| H4 | Variáveis demográficas do solicitante são utilizáveis | **MAJORITARIAMENTE INDISPONÍVEIS** | `verify_h3_prazo.py` |

## H1 — `AssuntoPedido` é saída da triagem

Medido no arquivo de 2026, cujas linhas recentes estão genuinamente sem triagem
(a data do retrato coincide com o `DataRegistro` mais novo). O arquivo de 2024
mostra apenas 0,098% de ausência e esconde o efeito por completo, porque toda
linha de 2024 já foi triada há cerca de dois anos.

| Janela antes do retrato | Linhas | `AssuntoPedido` ausente |
|---|---|---|
| últimos 3 d | 902 | **79,60%** |
| últimos 7 d | 3.189 | 57,98% |
| últimos 14 d | 6.018 | 49,47% |
| últimos 30 d | 13.811 | 32,46% |
| últimos 90 d | 41.299 | 12,41% |
| últimos 365 d | 113.538 | 4,64% |

Por situação: `Cadastrada` **59,5%** ausente contra `Concluída` **0,000%**.
Por resposta: não respondidos **58,2%** contra respondidos **0,000%**.

Decaimento monótono até zero conforme o pedido envelhece é a assinatura de um
campo preenchido retroativamente.

**Consequência:** um modelo treinado em pedidos históricos encerrados vê ~100%
de cobertura e encontraria ~60% de nulos em produção — distorção entre treino e
serviço (*train/serve skew*).

## H2 — `OrgaoDestinatario` é o órgão endereçado

Três testes independentes:

1. **Não há registros duplicados.** 655.177 linhas carregam 655.177 `IdPedido`
   distintos **e** 655.177 `ProtocoloPedido` distintos. O reencaminhamento nunca
   cria uma segunda linha.
2. **Junção com `Recursos`.** `Pedidos.OrgaoDestinatario == Recursos.OrgaoPedido`
   em **100,000%** dos casos, tanto reencaminhados quanto não (2024: n=10.089;
   2025: n=11.919). Isso prova consistência interna, **não** direção — os dois
   arquivos vêm do mesmo retrato.
3. **Teste direcional (decisivo).** Sob sobrescrita, órgãos que *expelem*
   pedidos mal endereçados mostrariam taxas *baixas*, e os absorvedores, taxas
   *altas*. O observado é o oposto:

| Órgão | Pedidos | Taxa de reenc. |
|---|---|---|
| SGPR – Secretaria-Geral da Presidência | 1.645 | **50,58%** |
| GSI-PR – Gabinete de Segurança Institucional | 1.652 | 44,25% |
| CC-PR – Casa Civil | 6.086 | 42,56% |
| MGI – Ministério da Gestão | 11.940 | 22,79% |
| *90 universidades federais (média)* | — | **1,09%** |

Média dos órgãos centrais/Presidência **36,69%** contra **1,09%** dos órgãos de
competência estreita — diferença de **33,7×**. A Casa Civil não pode
plausivelmente ser o principal *destino* de pedidos reencaminhados; ela é para
onde o cidadão manda o que não consegue situar. Isso coincide com a premissa do
próprio Termo de Abertura.

Teste rejeitado, registrado por completude: o prefixo NUP do protocolo
(`ProtocoloPedido[:5]`) **não** identifica órgão — pureza ponderada 0,58.

## H3 — `prazo_dias` é vazamento (o achado importante)

`prazo_dias = PrazoAtendimento − DataRegistro` era a variável dominante do
modelo, com **40,5% do ganho**. É posterior à triagem:

| `FoiProrrogado` | n | média | mediana | p05 | p95 |
|---|---|---|---|---|---|
| Não | 518.948 | 21,59 | **21,0** | 20,0 | 25,0 |
| Sim | 136.228 | 31,29 | **31,0** | 21,0 | 39,0 |

A diferença de **+10 dias** na mediana é exatamente a prorrogação única
concedida pelo art. 11 §2 da
[Lei 12.527/2011](https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2011/lei/l12527.htm).
`PrazoAtendimento` é reescrito quando a prorrogação é concedida — depois da
chegada. Logo `prazo_dias` reimportava `FoiProrrogado`, que já estava na lista
de exclusão, pela porta dos fundos.

Custo do vazamento, medido treinando uma variante diagnóstica que o mantém:

| Partição | PR-AUC honesta | PR-AUC vazada | Inflação |
|---|---|---|---|
| validação 2025 | 0,2053 | 0,4318 | **+22,65 pp** |
| teste 2026 | 0,1602 | 0,3772 | **+21,70 pp** |
| teste 2026 maturado | 0,1723 | 0,3965 | **+22,42 pp** |

Precisão@5% no teste maturado: **24,44% honesta contra 43,25% vazada.** O
vazamento quase dobra o desempenho aparente.

## H4 — variáveis demográficas quase ausentes

| Campo | Ausente | Distintos |
|---|---|---|
| `Escolaridade` | **76,23%** | 6 |
| `Profissao` | **76,87%** | 15 |
| `Genero` | 70,67% | 3 |
| `TipoDemandante` | 16,91% | 2 |

`IdSolicitante == '0'` (anonimizado) em 110.744 linhas (16,90%).

A regra de abstenção do Termo de Abertura — não pontuar quando falta dado de
perfil — dispararia em **77,58%** de todos os pedidos, o que não constitui
produto de triagem viável.

A ausência também **não é informativa**: a taxa de reencaminhamento é 7,54%
quando o perfil está presente e 7,25% quando ausente.

## Censura à direita

O arquivo de 2026 é um retrato de 2026-09-14, então um pedido de setembro teve
dias, não meses, para ser reencaminhado. Confirmado: taxa de positivos **5,75%**
entre linhas registradas com ≥60 dias de antecedência do retrato, contra
**3,60%** nas mais recentes. As métricas são portanto reportadas tanto no teste
completo quanto no maturado (`MATURITY_DAYS = 60`).

## O achado principal: aprendizado de máquina apenas empata com uma tabela

Somente variáveis honestas de chegada. Linha de base: ordenar pela taxa
histórica suavizada de reencaminhamento por órgão, ajustada só nos anos de
treino — um único `groupby`.

| Teste 2026 maturado | ROC-AUC | PR-AUC | prec@1% | prec@5% | prec@10% |
|---|---|---|---|---|---|
| Consulta por órgão | 0,7434 | 0,1641 | 36,12% | **24,79%** | 17,79% |
| LightGBM (161 árvores) | 0,7471 | 0,1723 | 36,59% | 24,44% | **18,70%** |

O modelo ganha 0,008 em PR-AUC e 0,9 pp em precisão@10%, e **perde em
precisão@5%**. 76% do seu ganho é identidade do órgão (`orgao_rate` 48,1% +
`OrgaoDestinatario` 27,7%).

**Conclusão:** removido o vazamento, praticamente todo o sinal recuperável é
"alguns órgãos são cronicamente mal endereçados". Isso ainda é operacionalmente
útil — **ganho de 4,3× na fila dos 5%** — mas não exige aprendizado de máquina,
e qualquer publicação precisa dizê-lo.
