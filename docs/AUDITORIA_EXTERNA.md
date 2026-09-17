# Auditoria externa — 17/09/2026

Auditoria independente do código e da documentação, executada por outro agente,
com a pauta em `/mnt/d/wsl-disks/auditoria-pauta.md` e log integral preservado em
`/mnt/d/wsl-disks/auditoria-codex-20260917.log` (9.384 linhas).

**Auditor:** `codex exec` (codex-cli 0.146.0), sandbox aberto, com acesso aos
1,1 GB de dados locais para conferir números — não apenas ler código.
**Consumo:** 288.744 tokens. **Interrompido por cota** antes de gravar o
relatório final e antes de cobrir a dimensão 7 (contrato HTTP e seguibilidade do
README). Os achados abaixo foram extraídos da narrativa no log.

Deliberadamente não informei ao auditor nenhuma conclusão prévia: ele recebeu o
que o projeto **afirma** ser, com instrução de verificar cada afirmação contra o
código e os dados. Auditoria que chega com as respostas não é auditoria.

## Achados

| # | Achado | Severidade | Situação |
|---|---|---|---|
| 1 | Vazamento do mesmo dia nos contadores por solicitante | **GRAVE** | declarado; correção planejada (Fix 1+2) |
| 2 | Variáveis consomem desfecho mais novo que `MATURITY_DAYS` | **GRAVE** | declarado; correção planejada (Fix 1+2) |
| 3 | Seleção de variáveis feita no conjunto de teste | **GRAVE** | pendente (Fix 3) |
| 4 | Precisão@k dependia da ordem das linhas | **MÉDIO** | **corrigido** |
| 5 | `Solicitantes` pode ser retrato atual, não perfil na abertura | **MÉDIO** | pendente (Fix 5, H6) |
| 6 | Números obsoletos na documentação | **MENOR** | pendente (Fix 6) |
| 7 | `historico_informado` mente sob histórico parcial | **MÉDIO** | pendente (Fix 7) |
| — | Reprodutibilidade | **positivo** | confirmada por terceiro |

### 1 — Vazamento do mesmo dia (GRAVE)

**159.320 linhas** recebem "histórico anterior" de outro pedido do mesmo
solicitante **no mesmo dia**, e **22.793** incorporam um rótulo positivo desse
empate.

Causa: `DataRegistro` é somente data, sem hora. `cumcount`/`cumsum` sobre o
quadro ordenado por data incluem os irmãos do mesmo dia, cuja ordem relativa é a
ordem do arquivo, não a temporal.

É uma **inconsistência interna**: as janelas por órgão usam
`searchsorted(..., side="left")`, que exclui o mesmo dia corretamente. O rigor
existia num lado e a documentação o generalizou para o outro.

Medição adicional feita depois: `IdPedido` **é** ordem de registro válida —
correlação **+0,996** com a data, apenas **22 inversões em 655.177 (0,0034%)**.
E **38% das linhas com solicitante real** estão em lotes do mesmo dia, com o
maior lote chegando a **370 pedidos**. Logo não se pode simplesmente descartar o
mesmo dia sem custo.

### 2 — Maturação violada pelas próprias variáveis (GRAVE)

**53.434 linhas** usam ao menos um desfecho de pedido registrado há menos de 60
dias; **523.719 de 654.718** taxas móveis por órgão incorporam algum positivo
com menos de 60 dias.

O projeto define `MATURITY_DAYS = 60` precisamente porque um rótulo precisa
desse tempo para ser confiável — e então deixa as variáveis consumirem rótulos
mais novos. No retrato de treino o desfecho já está registrado; em produção,
não estaria. É distorção treino/serviço na variável que vale 16% do ganho.

`docs/VERIFICATION.md` documentava a censura à direita **do rótulo**, o que fazia
parecer o assunto resolvido. O problema é a censura das **variáveis**, distinta e
não tratada.

**Restrição relevante:** a solução estatisticamente superior seria definir o alvo
num horizonte fixo ("reencaminhado em até 30 dias"), mas os dados **não têm data
de reencaminhamento**, só o booleano. Resta defasar as variáveis.

### 3 — Seleção de variáveis no conjunto de teste (GRAVE)

`scripts/experiment_features_v2.py`, na montagem da combinação final, escolhe os
grupos por `test_pr` — desempenho em 2026 — e em seguida anuncia o ganho **no
mesmo 2026**. Deveria selecionar por `val_pr`.

Consequência: o ganho publicado sobre a linha de base **não é estimativa
honesta de generalização**. A direção provavelmente se mantém; a magnitude está
otimista.

### 4 — Precisão@k dependia da ordem das linhas (MÉDIO) — CORRIGIDO

A implementação usava `np.argsort(-p)[:k]`, que desempata pela ordem das linhas
no arquivo. O escore da linha de base é a taxa por órgão, **constante dentro de
cada órgão**, logo há blocos enormes de empate no ponto de corte.

A dimensão do problema, agora que a função reporta o diagnóstico:

| Escore, fila | k | empatados no corte | usados do empate |
|---|---|---|---|
| base, validação top 1% | 1.501 | 1.491 | **1.291 (86%)** |
| base, teste top 10% | 8.610 | 3.075 | 2.868 (33%) |
| modelo, teste top 1% | 861 | 1 | 1 |
| modelo, teste top 5% | 4.305 | 1 | 1 |

Ou seja: **86% da fila de 1% da linha de base era sorteio**, não seleção. O
número nunca foi uma quantidade bem definida. Os escores do modelo, ao
contrário, são praticamente únicos, e por isso a correção quase não o move.

**Correção aplicada:** `precision_at_k` devolve o **valor esperado** sob
desempate uniforme — positivos estritamente acima do corte, mais a fração
proporcional do bloco empatado. Determinístico e independente de ordenação.
Coberto por `scripts/test_metrics.py`, inclusive um teste de 60 permutações que
exige espalhamento zero.

Efeito nos números: o modelo não muda; a base move-se em centésimos
(precisão@5% no teste maturado 0,2479 → 0,2480). **O ganho está na
defensibilidade, não no valor.**

### 5 — `Solicitantes` pode ser retrato atual (MÉDIO, não verificado)

**22.963 pessoas** aparecem em anos diferentes com **todos os campos
demográficos idênticos** — compatível com perfil de hoje replicado sobre pedidos
antigos. Se confirmado, `Escolaridade` e `Profissao` são posteriores nas linhas
velhas: alguém que se formou em 2024 apareceria formado em seus pedidos de 2022.

O auditor tentou resolver pela documentação da CGU e por busca web, sem
conclusão, e registrou como não verificável — comportamento correto.

Seria a **sexta família de vazamento** do projeto. Fica como H6.

### 6 — Números obsoletos na documentação (MENOR)

O que reproduz hoje é **31 variáveis / 52 árvores**; o `README.md` dizia 30/98 e
`docs/TREINAMENTO.md` dizia 20/161.

### 7 — `historico_informado` mente sob histórico parcial (MÉDIO)

Enviando apenas `n_pedidos_previos=80`, o serviço devolve
`historico_informado: true` com as outras seis variáveis em `-1` — combinação
que **nunca ocorre no treinamento**, onde 80 pedidos anteriores implicam razão
real. O modelo recebe uma linha fora da distribuição e o campo de transparência
afirma completude que não há.

### Achado positivo — reprodutibilidade confirmada

O treinamento oficial, executado pelo auditor, reproduziu os três arquivos de
modelo com **hashes idênticos**, mesmas métricas e mesmo limiar. Só o JSON
difere, porque grava hora corrente e tempos medidos — comportamento correto.

## O que a auditoria não cobriu

- **Dimensão 7 da pauta:** contrato HTTP e seguibilidade do README por um
  iniciante. Morreu por cota antes disso.
- **Relatório em linguagem acessível**, que era parte do pedido.

Previsto: repetir com `gemini` 0.59.0 — terceiro fornecedor, independente —
**depois** que as correções estabilizarem os números. Auditar documentação que
vai mudar seria desperdício.

## Nota metodológica

`docs/CAMPOS_POST_HOC.md` traz um checklist cujo item 4 é *"audite as derivadas,
não só os campos brutos"* e o item 9 é *"desconfie da sua melhor variável"*.
As duas regras foram escritas neste repositório e violadas neste repositório: as
derivadas de histórico não passaram pelo protocolo que o próprio documento
prescreve.

Isso é material de publicação, não vergonha. Um protocolo de auditoria de
vazamento cuja aplicação externa encontra três defeitos graves no trabalho que o
propõe é evidência de que o protocolo é necessário — inclusive contra quem o
escreveu. Ver [`LITERATURE.md`](LITERATURE.md), contribuição 2.

---

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
