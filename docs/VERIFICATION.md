# Auditoria de vazamento e viabilidade — risco de reencaminhamento LAI

Toda afirmação aqui é reproduzível a partir de `scripts/` contra o retrato
**20260914** dos dados abertos do Fala.BR. Fonte:
<https://dadosabertos-download.cgu.gov.br/FalaBR/Arquivos_FalaBR/>

Coorte: `Pedidos_csv_{2022..2026}`, **655.177 linhas**, taxa-base de
reencaminhamento **7,32%**. As 459 linhas com
`Situacao == "Encaminhada por Outro Órgão"` são descartadas da modelagem: estão
em trânsito, logo seu `OrgaoDestinatario` é o receptor, não o endereçado.

## AVISO — números desta página anteriores ao Fix 1+2

Auditoria externa (17/09/2026) mostrou que as variáveis de histórico do
solicitante sofriam **vazamento do mesmo dia** e **consumo de desfecho
imaturo**. Corrigidos os dois, o ganho do modelo sobre a linha de base
**desapareceu**: precisão@5% no teste maturado passou de 30,99% para **24,67%**,
contra **24,80%** da consulta por órgão.

Toda seção abaixo que atribua ganho ao histórico do solicitante (G2, T1) está
medida sobre variáveis vazadas e será republicada. As seções H1 a H5, que tratam
de exclusão de campos, permanecem válidas. Ver
[`AUDITORIA_EXTERNA.md`](AUDITORIA_EXTERNA.md).

## Resumo dos achados

| ID | Hipótese | Veredito | Script |
|----|-----------|---------|--------|
| H1 | `AssuntoPedido` é atribuído durante a triagem, não na chegada | **CONFIRMADA — excluído** | `verify_leakage_v2.py` |
| H2 | `OrgaoDestinatario` é sobrescrito no reencaminhamento | **REFUTADA — mantido** | `verify_h2_final.py`, `verify_h2_baserate.py` |
| H3 | `prazo_dias` é variável de chegada | **REFUTADA — é vazamento, excluído** | `verify_h3_prazo.py` |
| H4 | Variáveis demográficas do solicitante são utilizáveis | **MAJORITARIAMENTE INDISPONÍVEIS** | `verify_h3_prazo.py` |
| H5 | `protocolo_seq` (fatia do `ProtocoloPedido`) é variável de chegada | **REFUTADA — é vazamento, em quarentena** | `verify_h5_protocolo.py` |
| H6 | `Solicitantes` traz o perfil na abertura do pedido | **REFUTADA — é retrato atual; demográficas removidas** | `verify_h6_solicitantes.py` |

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

**Conclusão parcial:** com o conjunto original de variáveis, praticamente todo o
sinal recuperável era "alguns órgãos são cronicamente mal endereçados" — útil,
mas não exigia aprendizado de máquina.

## Atualização: o histórico do solicitante quebra o empate

`scripts/experiment_features_hpo.py` testou seis grupos de variáveis derivadas.
Cinco não produziram nada. Um mudou a conclusão.

**G2 — histórico do solicitante**, duas variáveis com acumulação deslocada e
ordenada por data.

> **Correção posterior:** auditoria externa mostrou que "estritamente causal",
> como estava escrito aqui, é falso. A acumulação impede ver a si mesma e o
> futuro, mas inclui pedidos do **mesmo dia** (159.320 linhas; 22.793 com
> rótulo positivo) e consome desfechos com menos de 60 dias (53.434 linhas).
> Os ganhos abaixo estão **otimistas**. Ver
> [`AUDITORIA_EXTERNA.md`](AUDITORIA_EXTERNA.md).

- `n_pedidos_previos` — quantos pedidos aquele `IdSolicitante` já fez antes
- `prev_reenc_solicitante` — quantos deles foram reencaminhados

| Variante | nvar | val PR-AUC | teste PR-AUC | teste prec@5% | teste AUC | Δ PR-AUC |
|---|---|---|---|---|---|---|
| base (chegada) | 20 | 0,2058 | 0,1738 | 0,2472 | 0,7471 | — |
| **+ G2 histórico** | **22** | **0,2305** | **0,2025** | **0,2736** | **0,7762** | **+0,0287** |
| + G1 ausência | 25 | 0,2048 | 0,1740 | 0,2462 | 0,7465 | +0,0002 |
| + G3 volume do órgão | 21 | 0,2045 | 0,1729 | 0,2455 | 0,7455 | −0,0009 |
| + G4 codificações de alvo extra | 23 | 0,2018 | 0,1736 | 0,2469 | 0,7469 | −0,0002 |
| + G5 calendário | 22 | 0,2053 | 0,1731 | 0,2448 | 0,7455 | −0,0007 |
| + G6 geografia | 21 | 0,2053 | 0,1710 | 0,2437 | 0,7464 | −0,0028 |
| todos os grupos | 34 | 0,2247 | 0,1968 | 0,2653 | 0,7727 | +0,0230 |

Ganho relativo de **+16,5% em PR-AUC** no teste maturado, com **duas** variáveis.
Juntar todos os grupos é **pior** que G2 sozinho (0,1968 contra 0,2025): os
outros cinco só acrescentam ruído.

**Isto supera a linha de base pela primeira vez.** Precisão@5% de **27,36%**
contra **24,79%** da consulta por órgão — +2,6 pp, ~10% relativo. Portanto o
aprendizado de máquina passa a ter justificativa, coisa que não tinha antes.

O mecanismo é interpretável e visível no dado bruto:

| Pedidos anteriores do solicitante | n | Taxa de reenc. |
|---|---|---|
| anônimo (`IdSolicitante == '0'`) | 110.718 | 7,68% |
| 0 (primeira vez) | 240.387 | **8,33%** |
| 1–2 | 77.776 | 7,67% |
| 3–10 | 74.815 | 6,48% |
| 11–50 | 75.125 | 5,46% |
| 50+ | 75.897 | **5,35%** |

Monotônico e decrescente: **solicitantes experientes aprendem qual órgão
endereçar.** Quem pede pela primeira vez tem 56% mais chance de ser
reencaminhado que quem já pediu mais de 50 vezes. Esse é um achado substantivo
sobre a LAI, não apenas uma variável útil.

**Ressalva de implantação:** as duas variáveis exigem estado por solicitante em
tempo de inferência, o que o serviço atual — deliberadamente sem estado — não
tem. Adotá-las obriga a manter um contador por `IdSolicitante`. A decisão é de
arquitetura, não de modelagem.

## H5 — `protocolo_seq` é vazamento (terceiro caso)

Na segunda rodada de engenharia de variáveis (`experiment_features_v2.py`), a
fatia `ProtocoloPedido[5:11]` elevou a PR-AUC de teste de 0,2018 para **0,4595**
— salto de 128% a partir de **uma única variável derivada de um identificador**.
Pelo protocolo de [`CAMPOS_POST_HOC.md`](CAMPOS_POST_HOC.md), isso é suspeita,
não comemoração.

`verify_h5_protocolo.py` refutou a variável em dois passos.

**A presunção de formato estava errada.** Supusemos NUP
`OOOOO SSSSSS AAAA DD`, com `[0:5]` identificando o órgão. Mas há **1.166
prefixos `[0:5]` distintos para 867 órgãos**, com **mediana de 23 órgãos por
prefixo** (máximo 219). O prefixo não é código de órgão, logo `[5:11]` não é um
sequencial neutro.

**Separação dentro do mesmo órgão e ano, que nenhuma variável legítima de
chegada produziria:**

| Órgão-ano | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| INSS 2022 (n=12.146) | **28,38%** | 15,25% | 12,55% | **0,36%** |
| INSS 2023 (n=9.898) | **30,67%** | 2,10% | 0,57% | 1,13% |
| ANVISA 2023 (n=7.004) | 5,60% | 1,88% | 1,60% | 1,94% |

Razão de **79×** entre o primeiro e o último quartil num único órgão-ano. A
correlação com a data é de apenas +0,23 a +0,28, então não é efeito temporal.
A leitura compatível com todas as evidências é que a fatia codifica a **unidade
administrativa registradora**, cujo comportamento de encaminhamento é quase
determinístico — informação sobre o canal, não sobre o pedido.

Mantida em quarentena explícita no código (`QUARENTENA = {"T3_protocolo"}`) para
que ninguém a reintroduza por achá-la promissora.

## Resultado final da engenharia de variáveis

Segunda rodada, nove grupos, cada um somado a base+G2. Sete não produziram nada.

| Grupo | nvar | val PR | teste PR | teste p@5% | AUC | Δ vs G2 |
|---|---|---|---|---|---|---|
| base (chegada) | 20 | 0,2058 | 0,1738 | 0,2472 | 0,7471 | −0,0280 |
| base + G2 (referência) | 22 | 0,2292 | 0,2018 | 0,2711 | 0,7758 | — |
| **T1_experiencia** | 27 | 0,2687 | **0,2452** | **0,3113** | 0,7926 | **+0,0434** |
| **BONUS_rate_movel** | 24 | 0,2408 | 0,2074 | 0,2760 | 0,7852 | **+0,0056** |
| T2_idade_orgao | 23 | 0,2342 | 0,2022 | 0,2623 | 0,7794 | +0,0004 |
| T3_geo (`mesma_regiao`) | 23 | 0,2292 | 0,2018 | 0,2711 | 0,7758 | 0,0000 |
| T2_volume_movel | 23 | 0,2297 | 0,1979 | 0,2657 | 0,7758 | −0,0039 |
| T2_tendencia | 23 | 0,2266 | 0,1969 | 0,2627 | 0,7756 | −0,0049 |
| T3_interacoes | 25 | 0,2252 | 0,1953 | 0,2708 | 0,7673 | −0,0065 |
| T3_municipio_p5 | 23 | 0,2250 | 0,1933 | 0,2697 | 0,7733 | −0,0085 |
| ~~T3_protocolo~~ | 23 | 0,4988 | 0,4595 | 0,4827 | 0,8568 | **vazamento (H5)** |
| **COMBINAÇÃO honesta** | **30** | **0,2784** | **0,2542** | **0,3069** | **0,8003** | **+0,0524** |

**Contra a linha de base sem modelo, no teste maturado de 2026:**

| | PR-AUC | precisão@5% |
|---|---|---|
| Consulta histórica por órgão | 0,1641 | 0,2479 |
| **Modelo final (30 variáveis)** | **0,2542** | **0,3069** |
| Ganho relativo | **+55%** | **+23,8%** |

A conclusão inicial de que o aprendizado de máquina apenas empatava com uma
tabela **não vale mais**. Com histórico do solicitante e taxa móvel por órgão, o
modelo supera a consulta em 5,9 pontos percentuais de precisão@5%.

Ganho por variável na combinação final:

| Variável | % do ganho |
|---|---|
| `orgao_rate` | 38,14 |
| **`orgao_rate_movel_90d`** | **19,95** |
| `OrgaoDestinatario` | 11,03 |
| **`n_pedidos_previos_neste_orgao`** | **6,67** |
| `Municipio_sol` | 5,23 |
| `prev_reenc_rate_solicitante` | 4,79 |
| `prev_reenc_neste_orgao` | 3,82 |
| `orgao_rate_movel_365d` | 2,06 |
| `dias_desde_primeiro_pedido_do_orgao` | 1,48 |
| `dias_desde_ultimo_pedido` | 1,39 |
| `n_pedidos_previos` (agregado) | 1,29 |
| `n_orgaos_distintos_previos` | 0,97 |

Dois achados substantivos aqui. Primeiro, **a experiência é específica do
órgão**: `n_pedidos_previos_neste_orgao` vale 6,67% contra 1,29% do agregado
`n_pedidos_previos` — cinco vezes mais. Conhecer o INSS não ajuda a endereçar o
MGI. Segundo, **a taxa móvel de 90 dias vale mais que qualquer variável de
solicitante** (19,95%): a tabela estática ajustada em 2022–2024 estava
desatualizada, como a queda da taxa-base de 8,03% para 5,23% sugeria.

Curiosidade metodológica: `orgao_rate_tendencia` e `orgao_volume_movel`, isolados,
foram **negativos**, mas os *níveis* móveis dos quais derivam são fortemente
positivos. Diferença de duas estimativas ruidosas é mais ruído.

## Idade do órgão — hipótese confirmada

Com a datação corrigida (2012-05-15 a 2026-09-01; a versão anterior estava
censurada em 2022 porque a CGU trocou o retrato de `20260914` para `20260915` no
meio do trabalho e um prefixo fixo perdia os arquivos antigos em silêncio):

| Idade do órgão no registro | n | Taxa de reenc. |
|---|---|---|
| < 1 ano | 6.182 | **9,66%** |
| 1–2 anos | 9.298 | **9,81%** |
| 2–4 anos | 22.471 | **9,85%** |
| 4–8 anos | 31.397 | 6,72% |
| 8+ anos | 324.958 | **5,94%** |

Órgãos com menos de quatro anos encaminham a ~9,8%; os com mais de oito, a
5,94% — razão de **1,65×**, com degrau nítido em torno de quatro anos. Órgão
nascido de reorganização administrativa tem fronteira de competência obscura, e
o cidadão erra mais. Como variável marginal rende pouco (+0,0004), porque
`orgao_rate` já absorve o efeito, mas é explicação causal publicável.

## H6 — `Solicitantes` é um retrato atual, não o perfil na abertura

Duas previsões opostas: se o cadastro guardasse o perfil de cada pedido, ao
longo de cinco anos **alguém** teria de mudar de escolaridade ou profissão; se
fosse um retrato replicado, a variação seria zero.

| Campo | Pessoas com mais de um valor entre 2022 e 2026 |
|---|---|
| `Escolaridade` | **0** |
| `Profissao` | **0** |
| `Genero`, `UF`, `Municipio`, `Pais`, `TipoDemandante`, `TipoPessoaJuridica` | **0** |
| `DataNascimento` (controle) | 0 |

22.963 pessoas aparecem em mais de um ano. **Nenhum campo muda em nenhuma
delas.** E das 1.999 presentes tanto em 2022 como em 2026, **1.999 (100%)** têm
registro idêntico nos nove campos. O arquivo anual é o mesmo cadastro recortado.

**Isto não é vazamento de alvo.** O modelo não passa a ler o rótulo, como
acontecia com `prazo_dias`. É **descasamento de tempo de medição nas
covariáveis**: as linhas antigas carregam um perfil futuro. Prejudica o treino,
não o serviço — onde o perfil corrente é o correto.

**Decisão: removidas da produção.** Custo medido: PR-AUC do teste maturado
0,1836 com elas contra **0,1853** sem; precisão@5% 24,67% contra 24,34%. Tudo
ruído. Com custo nulo e defeito metodológico real, remover é o correto.

Efeito na auditoria de equidade, que melhorou: a razão de `Ensino Fundamental`
na fila de 10% caiu de **2,40×** para **1,20×** da participação populacional.
Parte da disparidade anterior vinha de o modelo usar a escolaridade diretamente.
