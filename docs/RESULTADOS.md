# Resultados — e por que não confiar no escore sem ler isto

O [`README.md`](../README.md) é o **manual de uso**: instala, sobe o serviço,
faz a primeira chamada. Este arquivo é a **avaliação**: diz se o escore presta.

Todo número de desempenho vem de [`METRICAS.md`](METRICAS.md), que é **gerado**
pelo treinamento. Os relatórios de auditoria estão em
[`auditorias/`](auditorias/INDICE.md).

## O resultado, em uma frase

**O modelo não supera uma consulta à taxa histórica do órgão** — que é uma
tabela de uma linha — na métrica que o produto entrega.

| Escore, teste 2026 maturado | PR-AUC | prec@5% |
|---|---|---|
| Consulta histórica por órgão (sem modelo) | 0,1641 | **24,80%** |
| LightGBM | 0,1901 | 24,23% |

Perde **2,3%** em precisão@5%, que é a fila que o produto de fato entrega.
Ganha 15,8% em PR-AUC e 9,7% em precisão@10%.

A distância encolheu muito com a correção do H10 — era −12,1% —, mas o sinal
não mudou: na métrica que o produto entrega, **a tabela de consulta continua à
frente**.

Conjunto de teste 2026 maturado (registrados com ao menos 60 dias de
antecedência do retrato), treino em 2022–2024, validação em 2025, taxa-base
5,75%.

> **Estes números de conclusão são mantidos à mão.** O
> `scripts/atualiza_numeros_docs.py` regenera a tabela de ganho abaixo, os
> blocos de resposta do README e os decimais de seis casas. Este parágrafo usa
> quatro casas e fica fora do alcance dele e do `check_prosa.py` — foi aqui que
> morou um "84,8%" errado durante dois dias. Ao retreinar, confira contra
> [`METRICAS.md`](METRICAS.md) à mão.

## Por que o modelo é, essencialmente, a tabela de consulta

Somando o ganho das quatro variáveis que são identidade do órgão —
`orgao_rate`, `OrgaoDestinatario` e as duas taxas móveis — chega-se a **77,36%**.
Contando também a idade do órgão, **79,72%**. Todo o histórico do solicitante
soma **10,09%**.

Ou seja: o modelo aprende, sobretudo, *quais órgãos são cronicamente mal
endereçados*. E isso uma tabela de consulta já sabia.

### Ganho por variável, medido no artefato de produção

| Variável | Ganho |
|---|---|
| `orgao_rate` | 44,93% |
| `OrgaoDestinatario` | 13,86% |
| `orgao_rate_movel_90d` | 10,97% |
| `Municipio_sol` | 5,68% |
| `orgao_rate_movel_365d` | 7,60% |
| `dias_desde_primeiro_pedido_do_orgao` | 2,37% |
| `prev_reenc_rate_neste_orgao` | 2,89% |
| `n_pedidos_previos_neste_orgao` | 2,44% |
| `n_pedidos_previos` | 1,20% |
| `n_orgaos_distintos_previos` | 0,92% |
| `prev_reenc_rate_solicitante` | 1,34% |
| `dias_desde_ultimo_pedido` | 0,87% |
| `idade` | 0,38% |
| `reg_month` | 0,69% |
| `reg_day` | 0,26% |
| `OrigemSolicitacao` | 0,61% |
| `reg_dow` | 0,40% |
| `Profissao` | 0,25% |
| `prev_reenc_solicitante` | 0,25% |
| `Municipio` | 0,47% |
| `UF_sol` | 0,14% |
| `UF` | 0,66% |
| `Genero` | 0,17% |
| `TipoDemandante` | 0,21% |
| `prev_reenc_neste_orgao` | 0,17% |
| `TipoPessoaJuridica` | 0,09% |
| `Esfera` | 0,06% |
| `Escolaridade` | 0,07% |
| `Pais` | 0,03% |
| `FormaResposta` | 0,01% |
| `uf_match` | 0,00% |

Ganho é peso no ajuste das árvores, **não** causalidade: diz o quanto a variável
foi usada para separar os casos, não que ela cause o reencaminhamento.

## Este resultado já foi muito melhor, e era vazamento

Versões anteriores anunciavam **+26,4%** sobre a linha de base, com precisão@5%
de 30,99%. Auditoria externa independente mostrou que o ganho vinha de defeitos
nas variáveis de histórico:

- **vazamento do mesmo dia:** 159.320 linhas recebiam histórico de um pedido do
  mesmo solicitante no mesmo dia, 22.793 delas com rótulo positivo;
- **desfecho imaturo:** 53.434 linhas consumiam resultado de pedido com menos de
  60 dias — desfecho que em produção ainda não seria conhecido.

Corrigidos os dois, o ganho desapareceu. Depois vieram mais quatro defeitos.
Os três primeiros (H7, H8, H9) **aumentaram** a distância para a linha de base,
de −0,5% para −4,3% e daí para −12,1%. O quarto foi na direção oposta: o
**H10** — a codificação por órgão não usava cross-fitting, de modo que cada
linha de treino carregava o próprio rótulo na maior variável do modelo.
Corrigido, a distância encolheu para os **−2,3%** de hoje, e o modelo passou de
160 para 56 árvores: ele não precisava mais daquela profundidade toda para
explorar uma variável otimista. O histórico completo, com o que
cada auditoria encontrou, está em [`auditorias/`](auditorias/INDICE.md).

**A conclusão original do projeto volta a valer**, agora medida sobre variáveis
honestas: praticamente todo o sinal recuperável é "alguns órgãos são
cronicamente mal endereçados".

## A reviravolta das janelas móveis

Durante dois dias este projeto afirmou que a defasagem de maturação havia
esvaziado a janela de 90 dias (2,81% do ganho) em favor da de 365 (10,20%).
**Era artefato de um defeito.** O prior de suavização vinha da taxa-base de todo
o período, 2025 e 2026 inclusive — vazamento H8.

Com o prior honesto a ordem se inverte: a janela de **90 dias vale 10,97%** e a
de 365 vale 7,60%. A janela curta é a que informa; era o vazamento que a fazia
parecer inútil.

## As variáveis demográficas, e a limitação H6

O modelo **usa** escolaridade, profissão, gênero e residência do solicitante,
porque o Termo de Abertura as inclui explicitamente no escopo.

A limitação **H6**: a tabela `Solicitantes` da CGU é um **retrato de hoje**, não
o perfil de quando o pedido foi feito. A prova é direta — das 22.963 pessoas que
aparecem em mais de um ano, **nenhuma** muda de escolaridade ou profissão em
cinco anos, e 100% dos registros são idênticos entre 2022 e 2026. Um pedido de
2022 carrega o perfil de 2026.

Testamos se isso prejudica o modelo. **Não prejudica de forma mensurável:** o
ganho das demográficas é até *maior* no treino de 2022 (+0,0121 de PR-AUC), onde
o retrato está mais defasado, do que no de 2024 (+0,0052). Sem tendência, sem
contaminação detectável. Remover as variáveis também não mudaria nada — na
medição de 18/09/2026 o modelo ficava marginalmente **melhor** sem elas (PR-AUC
0,1883 contra 0,1855), diferença dentro do ruído.

> Os números por ano (+0,0121 e +0,0052) e o contrafactual foram medidos
> **antes** das correções de H7, H8 e H9, e ainda não foram remedidos. A
> conclusão qualitativa não depende deles; os valores exatos vão mudar.
> Detalhes em `scripts/experiment_h6_mitigacao.py`.

O dado pessoal que o serviço recebe — perfil e contadores de histórico — **não
é retido no artefato**. Ver
[`DECISAO_ESTADO_SOLICITANTE.md`](DECISAO_ESTADO_SOLICITANTE.md).

## Por que o chamador manda numerador e denominador, e não a razão pronta

O denominador conta **só** pedidos cujo desfecho já maturou (60 dias). Enviar a
razão pronta deixaria o chamador escolher, sem saber, um denominador não
maturado — e reintroduziria pelo contrato exatamente o vazamento que a correção
de maturação eliminou no treinamento.

## Limitações conhecidas

- **`Escolaridade` 76,2% ausente**, `Profissao` 76,9%. A regra de abstenção do
  Termo de Abertura — não pontuar quando falta perfil — recusaria **77,58%** dos
  pedidos, o que não é um produto viável.
- Variáveis demográficas somam pouco; **77,36% do ganho é identidade do órgão**,
  e elas vêm de um retrato atual do cadastro (H6, acima).
- Rótulos de 2026 sofrem **censura à direita** (5,75% de positivos entre os
  maturados contra 3,60% nos recentes).
- As tabelas por órgão são um retrato do fim da janela de dados; exigem reajuste
  periódico, sem o qual `orgao_rate_movel_90d` envelhece e perde valor.
- **45,8% das linhas não têm histórico de solicitante aproveitável** (16,9%
  anonimizadas, 28,9% de quem pediu uma única vez), então o ganho do histórico
  vem de pouco mais da metade do volume.
- O limiar é ponto de operação da fila de 10%, **não** probabilidade calibrada, e
  muda a cada retreinamento. Valor corrente em [`METRICAS.md`](METRICAS.md).
- O ganho por variável acima é regenerado por
  `scripts/atualiza_numeros_docs.py` e conferido por `scripts/check_prosa.py`;
  o parágrafo de conclusão, não.
