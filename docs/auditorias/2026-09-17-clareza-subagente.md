> Registro histórico, copiado sem alteração de `docs/AUDITORIA_EXTERNA.md`
> quando aquele arquivo foi dividido por data, em 18/09/2026. Os números
> aqui são os **daquele momento** e não devem ser lidos como correntes;
> para os correntes veja [`../METRICAS.md`](../METRICAS.md).

# Segunda auditoria — clareza para iniciante, 17/09/2026

**Auditor:** subagente com **contexto restrito**, somente leitura, persona de
aluno no meio do primeiro curso de aprendizado de máquina num curso tecnólogo de
Sistemas de Informação. Autorizado a ler **apenas** `README.md` e a executar o
tutorial literalmente.

A restrição é o método, não uma limitação: um auditor que nada sabe do projeto
não consegue trapacear com conhecimento que não tem. É o proxy mais fiel do
leitor real que se consegue construir.

Fecha a dimensão 7 da primeira auditoria, que morreu por cota antes de chegar lá.

## Veredito: 3 de 4 objetivos cumpridos

| Objetivo | Resultado |
|---|---|
| Achar o arquivo do modelo | **CONSEGUE** |
| Saber entradas e saídas | **CONSEGUE** |
| Colocar de pé e obter resposta | **travou** (causa externa, ver nota) |
| Saber que existem os documentos vizinhos | **CONSEGUE** |

Sobre o terceiro: o auditor bateu em `OSError: [Errno 98] Address already in
use`. A causa era um `bentoml serve` **órfão deixado por mim**, resíduo de uma
auditoria anterior — não defeito do tutorial. Verificado em ambiente limpo:
`/score` e `/health` respondem corretamente. A **ausência de tratamento** desse
erro no README, porém, era defeito real, e foi corrigida.

## Achado mais grave, e era meu

A seção "Sem dado demográfico algum" afirmava que o modelo não usa variáveis
demográficas — **falso** desde que elas foram restauradas ao escopo do TAP. Eu
introduzi a contradição ao remendar o README em camadas sem reler o conjunto.

## Os nove achados aplicados

1. Contradição das demográficas, reescrita.
2. Passo 1 clonava o repositório dentro de si mesmo para quem já tinha a pasta.
3. Passo 3 exibia `trees`/`features` fixos que nunca batem com a execução real.
4. Passo 4 sem tratamento de porta ocupada.
5. `protocolo_seq`, citado como um dos dois piores vazamentos, ausente da tabela
   de exclusões.
6. `AUDITORIA_EXTERNA.md` citado no corpo e ausente da tabela Estrutura.
7. **Glossário de 25 termos** — o achado mais valioso. O auditor listou dezenove
   termos usados sem definição: PR-AUC, precisão@k, ganho, calibrado, quantil,
   coorte, taxa-base, H1–H6, prior, censura à direita, vazamento, *post hoc*,
   BentoML, `uv`, artefato, featurização, ablação, HPO, pp. Observação
   particularmente aguda: o README **descreve** codificação de alvo sem nunca
   nomeá-la, então um iniciante não teria como pesquisar a técnica depois.
8. Resultado do cenário "sem histórico" afirmado sem mostrar o comando.
9. Formato de data só por exemplo, nunca como regra.

## Defeitos que a limpeza subsequente revelou

O remendo em camadas produziu: duplicação de `METRICAS.md` na mesma frase, dois
parágrafos de ganho por variável com números divergentes (84,8% correto contra
73,7% obsoleto), tabela de resultado com números do modelo sem demografia,
explicação do `uv sync` duplicada, e uma limitação citando "76% é identidade do
órgão".

Lição repetida pela terceira vez neste projeto: **informação duplicada divergE**.
Foi a causa dos números obsoletos, das três cópias de `precision_at_k`, do export
incoerente com o modelo, e agora destas duplicações. `METRICAS.md` gerado e
`check_docs_numbers.py` existem por isso.
