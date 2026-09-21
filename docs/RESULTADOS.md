# Resultados, e por que não confiar neles sem ler isto

Este documento é o **relatório de pesquisa** do projeto. O
[`README.md`](../README.md) é o tutorial: serve para pôr o modelo de pé. Aqui
estão o resultado, a história de como ele encolheu, e as limitações — é o que
justifica, ou não, usar o escore.

Todo número de desempenho vem de [`METRICAS.md`](METRICAS.md), que é **gerado**
pelo treinamento. Os relatórios de cada auditoria estão em
[`auditorias/`](auditorias/INDICE.md).

## O resultado, em uma frase

O modelo **não supera** uma consulta à taxa histórica do órgão — uma tabela de
uma linha — na métrica que o produto entrega, que é a precisão no topo da fila.
Perde 4,3% em precisão@5%. O quadro completo está no
[`README.md`](../README.md#resultado-principal) e em
[`METRICAS.md`](METRICAS.md).

> ### Este número já foi muito melhor, e era vazamento
>
> Versões anteriores deste README anunciavam **+26,4%** sobre a linha de base
> (precisão@5% de 30,99%). Auditoria externa independente mostrou que o ganho
> vinha de dois defeitos nas variáveis de histórico:
>
> - **vazamento do mesmo dia:** 159.320 linhas recebiam histórico de um pedido
>   do mesmo solicitante no mesmo dia, 22.793 com rótulo positivo;
> - **desfecho imaturo:** 53.434 linhas consumiam resultado de pedido com menos
>   de 60 dias, e 523.719 das taxas móveis incorporavam algum — desfecho que em
>   produção ainda não seria conhecido.
>
> Corrigidos os dois (defasagem de maturação nas variáveis de desfecho, ordem
> `(data, IdPedido)` nas de contagem), o ganho desapareceu. Registro completo em
> [`docs/auditorias/`](auditorias/INDICE.md).

**A conclusão original do projeto volta a valer:** com variáveis honestas de
chegada, praticamente todo o sinal recuperável é "alguns órgãos são
cronicamente mal endereçados", e uma tabela de consulta de uma linha captura
isso.

Sobre as duas janelas móveis, vale registrar uma reviravolta. Durante um tempo
este README dizia que a defasagem de 60 dias havia esvaziado a janela de 90 dias
(2,81% do ganho) em favor da de 365 (10,20%). **Era artefato de um defeito.** O
prior de suavização vinha da taxa-base de todo o período, 2025 e 2026 inclusive
— vazamento H8, corrigido em 18/09/2026. Com o prior honesto, a ordem se
inverte: a janela de 90 dias vale **9,16%** e a de 365 vale **7,02%**. A janela
curta é a que informa; era o vazamento que a fazia parecer inútil.


## As variáveis demográficas: usadas, com uma limitação declarada

O modelo **usa** escolaridade, profissão, gênero e residência do solicitante,
porque o Termo de Abertura do projeto as inclui explicitamente no escopo.

Há uma limitação conhecida, chamada **H6**: a tabela `Solicitantes` da CGU é um
**retrato de hoje**, não o perfil de quando o pedido foi feito. A prova é direta
— das 22.963 pessoas que aparecem em mais de um ano, **nenhuma** muda de
escolaridade ou profissão em cinco anos, e 100% dos registros são idênticos
entre 2022 e 2026. Ou seja, um pedido de 2022 carrega o perfil de 2026.

Testamos se isso prejudica o modelo. **Não prejudica de forma mensurável:** o
ganho das demográficas é até *maior* no treino de 2022 (+0,0121 de PR-AUC), onde
o retrato está mais defasado, do que no de 2024 (+0,0052). Sem tendência, sem
contaminação detectável. Remover as variáveis também não mudaria nada — na
verdade o modelo ficava marginalmente **melhor** sem elas (PR-AUC 0,1883 contra
0,1855 na medição de 18/09/2026), diferença dentro do ruído. Ainda não foi
remedido depois da correção de H9. Detalhes em
`scripts/experiment_h6_mitigacao.py`.

> Os dois números de ganho por ano acima (+0,0121 e +0,0052) foram medidos
> **antes** da correção de H7 e H8, em 17/09/2026, e ainda não foram
> remedidos. A conclusão qualitativa — sem tendência detectável — não depende
> deles, mas os valores exatos vão mudar.

O dado pessoal que o serviço recebe — perfil e contadores de histórico — **não é
retido no artefato**. Ver
[`docs/DECISAO_ESTADO_SOLICITANTE.md`](DECISAO_ESTADO_SOLICITANTE.md).


# Limitações conhecidas

- **`Escolaridade` 76,2% ausente**, `Profissao` 76,9%. A regra de abstenção do
  Termo de Abertura — não pontuar quando falta perfil — recusaria **77,58%** dos
  pedidos, o que não é um produto viável.
- Variáveis demográficas somam pouco; **72,14% do ganho é identidade do
  órgão**. E elas vêm de um retrato atual do cadastro, não do perfil na
  abertura do pedido — limitação H6, sem efeito mensurável medido.
- **O ganho por variável é mantido à mão neste README e já divergiu três
  vezes.** Os valores acima foram medidos em 18/09/2026 contra
  `artifacts/model_arrival.txt`. Gerá-los junto de
  [`docs/METRICAS.md`](METRICAS.md) é a correção estrutural pendente.
- Rótulos de 2026 sofrem **censura à direita** (5,75% de positivos entre os
  maturados contra 3,60% nos recentes).
- As tabelas por órgão são um retrato do fim da janela de dados; exigem reajuste
  periódico, sem o qual `orgao_rate_movel_90d` envelhece e perde valor.
- 45,8% das linhas não têm histórico de solicitante aproveitável (16,9%
  anonimizadas, 28,9% de quem pediu uma vez só), então o ganho vem de pouco
  mais da metade do volume.
- O limiar é ponto de operação da fila de 10%, não probabilidade calibrada, e
  muda a cada retreinamento. Valor corrente em
  [`docs/METRICAS.md`](METRICAS.md).

