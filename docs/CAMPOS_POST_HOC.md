# Campos *post hoc*: o que são, por que não servem para treinar, e como os identificamos

Documento conceitual e metodológico. O registro empírico dos achados está em
[`VERIFICATION.md`](VERIFICATION.md); aqui explicamos o **porquê** e o **como**.

---

## 1. Definição

Um **campo *post hoc*** é uma coluna cujo valor, no momento em que a decisão
precisa ser tomada, **ainda não existe** — ou existe com outro valor. Ele é
preenchido, corrigido ou reescrito *depois* daquele instante.

O nome usual na literatura é **vazamento de alvo** (*target leakage*) ou
**vazamento temporal**. É diferente de duas coisas com que se confunde:

| Não confundir com | Diferença |
|---|---|
| **Correlação espúria** | Uma correlação espúria é fraca ou instável. Um campo *post hoc* costuma ser **fortíssimo e estável** — é justamente isso que o torna perigoso |
| **Variável redundante** | Redundância prejudica interpretabilidade, não validade. Vazamento invalida a medição inteira |

A definição depende de **um instante de decisão declarado**. Sem ele a pergunta
"este campo é *post hoc*?" não tem resposta. Neste projeto o instante está
fixado pelo Termo de Abertura: *no recebimento do pedido na caixa de entrada do
Fala.BR, precedendo o encaminhamento interno primário*.

Por isso `AssuntoPedido` é *post hoc* aqui e **não seria** num projeto cujo
instante de decisão fosse "depois da triagem inicial". O campo não é
intrinsecamente ruim; ele é ruim **em relação a este instante**.

---

## 2. Por que não se pode treinar com eles — teoria

### 2.1 O modelo aprende a distribuição errada

Treinar é estimar `P(y | x)`. Servir é aplicar essa estimativa a `x` colhido na
chegada. Se um campo `x_j` tem uma distribuição no treino (histórico encerrado)
e outra no serviço (pedido recém-chegado), então

```
P_treino(y | x)  ≠  P_serviço(y | x)
```

e o modelo está resolvendo um problema que não é o seu. Isso se chama
**distorção treino/serviço** (*train/serve skew*). Nenhuma quantidade de
validação cruzada detecta o problema, porque **os dados de validação sofrem da
mesma distorção que os de treino**.

### 2.2 O sentido causal está invertido

No caso mais severo, o campo não é causa do alvo: é **consequência** dele, ou
consequência do mesmo evento que o produz. Em termos de grafo:

```
o desejado      x  ──►  y          (x precede e informa y)
o que ocorre    y  ──►  x_j        (x_j é efeito de y)
```

`FoiProrrogado` é o exemplo puro: um pedido é prorrogado *porque* deu trabalho
— e reencaminhamento é uma das razões de dar trabalho. Condicionar em
`FoiProrrogado` é condicionar num descendente do alvo. O modelo parece
excelente e não prediz nada; ele **lê a resposta**.

### 2.3 O horizonte informacional é violado

Todo problema de predição tem um **horizonte informacional**: o conjunto de
fatos conhecidos em `t = 0`. Usar `x_j` revelado em `t > 0` equivale a consultar
o futuro. A validação temporal (treinar no passado, testar no futuro) é
necessária mas **não suficiente**: ela protege contra vazamento entre *linhas*,
não contra vazamento entre *colunas*. Um campo *post hoc* atravessa um corte
temporal perfeito sem ser notado, porque o vazamento está dentro de cada linha.

### 2.4 Custo de oportunidade: o vazamento canibaliza o sinal legítimo

Um campo *post hoc* forte não apenas infla a métrica — ele **rouba** capacidade
do modelo. As árvores dividem primeiro pela variável mais informativa; se ela é
vazada, as variáveis honestas nunca são exploradas. Neste projeto, ao remover
`prazo_dias`, o ganho atribuído a `orgao_rate` subiu de 27,7% para **48,1%**: o
sinal legítimo estava lá o tempo todo, encoberto.

---

## 3. Por que não se pode treinar com eles — prática

O que acontece de fato, em ordem cronológica de um projeto real:

1. **O retrospecto parece ótimo.** Neste caso a PR-AUC era 0,3772. Aparência de
   modelo pronto para produção.
2. **A revisão técnica aprova.** O corte é temporal, a métrica é adequada, não
   há mistura entre treino e teste. Tudo o que uma revisão costuma checar está
   correto.
3. **Em produção o campo chega vazio.** Para `AssuntoPedido`, 79,6% de ausência
   em pedidos de 3 dias. O modelo recebe `NaN` onde treinou com valor.
4. **O desempenho colapsa** e o diagnóstico é caro, porque nada no código está
   errado: a falha está na *semântica dos dados*, não na engenharia.
5. **Custo institucional.** Num contexto de LAI, a fila prioritária passa a ser
   ruído, o analista sênior perde confiança na ferramenta e o prazo legal que se
   queria proteger é consumido de todo modo.

Números deste projeto, medidos e não estimados:

| Campo | Ganho aparente | Realizável em produção |
|---|---|---|
| `prazo_dias` | +21,70 pp de PR-AUC no teste; precisão@5% de 22,73% → 39,36% | **zero** — o valor muda após a prorrogação |
| `AssuntoPedido` | +1,31 pp na validação | **zero** — 79,6% ausente na chegada; e no teste fora do tempo o efeito é **−0,53 pp**, ou seja, piora |

O segundo caso merece atenção: um campo vazado **nem sempre ajuda**. Ele
introduziu dependência de um padrão de preenchimento que não se repete no ano
seguinte. Vazamento não é só otimismo — é também fragilidade.

---

## 4. Como identificamos os campos — protocolo detalhado

Não existe um teste único. Aplicamos quatro, e **cada campo exigiu um teste
diferente**. É este o aspecto metodologicamente transferível do trabalho.

### Passo 0 — declarar o instante de decisão

Sem isso nada é decidível. Fixado: chegada no Fala.BR, antes do encaminhamento
primário. Unidade de análise: um pedido.

### Passo 1 — triagem documental

Ler a documentação de campos da CGU e separar por tempo verbal declarado. Campos
descritos como *"em branco para pedidos que ainda estejam na situação Em
Tramitação"* (`DataResposta`, `Decisao`, `EspecificacaoDecisao`) são *post hoc*
por definição — nem precisam de teste.

**Limitação, e por isso não paramos aqui:** a documentação estava incompleta e
parcialmente errada. O esquema real tem **23 colunas**, três mais que o
documentado (`DetalhamentoDecisao`, `MotivoNegativaAcesso`,
`PrazoRestricaoAcesso`). E `DataRegistro` é documentado como
`DD/MM/AAAA HH:MM:SS` mas vem **somente com a data**.

### Passo 2 — teste de decaimento por recência (para `AssuntoPedido`)

**Ideia:** se um campo é preenchido durante a triagem, sua ausência deve
**decair monotonicamente** conforme o pedido envelhece. Um campo de chegada não
apresenta esse gradiente.

**Requisito crítico:** o teste só funciona no **ano corrente**. Rodado no arquivo
de 2024, `AssuntoPedido` aparece com 0,098% de ausência e o campo parece
perfeito — porque toda linha de 2024 já foi triada há dois anos. A primeira
rodada da nossa auditoria caiu exatamente nessa armadilha.

Repetido em 2026, cujo retrato coincide com o registro mais recente:

| Janela | Ausência |
|---|---|
| 3 dias | **79,60%** |
| 7 dias | 57,98% |
| 30 dias | 32,46% |
| 365 dias | 4,64% |

Confirmação cruzada por estado, que não depende de datas: `Cadastrada` 59,5%
ausente contra `Concluída` **0,000%**; sem resposta 58,2% contra respondidos
**0,000%**. Convergência para zero é a assinatura do preenchimento retroativo.

### Passo 3 — teste de assinatura estatutária (para `prazo_dias`)

**Ideia:** quando a regra de negócio é uma **norma escrita**, a norma prevê um
valor numérico exato. Se o dado exibe exatamente esse valor, o campo foi
recalculado sob a norma.

A LAI concede uma prorrogação de **+10 dias** (art. 11 §2). Medimos:

| `FoiProrrogado` | mediana de `prazo_dias` |
|---|---|
| Não | **21,0** |
| Sim | **31,0** |

Diferença de exatamente **+10 dias**. Não é correlação vaga: é a norma impressa
no dado. `PrazoAtendimento` é reescrito quando a prorrogação é concedida.

**Este é o achado mais importante do protocolo**, porque `prazo_dias`:

- não é um campo bruto, mas uma **variável derivada** que nós mesmos criamos;
- reimportou `FoiProrrogado`, que **já estava na lista de exclusão**;
- era a variável **mais forte** do modelo, com 40,5% do ganho.

**Lição geral: excluir o campo bruto não basta.** Toda derivada precisa ser
auditada contra a mesma pergunta, porque uma derivada pode reconstruir por
aritmética aquilo que a exclusão removeu. Aqui, `PrazoAtendimento − DataRegistro`
é uma função quase determinística de `FoiProrrogado`.

### Passo 4 — teste direcional (para `OrgaoDestinatario`)

Usado quando a suspeita não pode ser resolvida por ausência nem por aritmética,
porque **o campo está sempre preenchido** e a dúvida é se o *valor* foi
trocado. Não há retrato anterior para comparar.

**Ideia:** formular as duas hipóteses como **previsões opostas e observáveis**, e
ver qual o mundo satisfaz.

| Hipótese | Previsão |
|---|---|
| Campo reescrito para o receptor | Órgãos que expelem pedidos mal endereçados exibem taxa **baixa**; absorvedores exibem taxa **alta** |
| Campo mantém o endereçado | Órgãos que o cidadão erra exibem taxa **alta**; órgãos de competência estreita ficam perto de **0%** |

Observado: Presidência e órgãos centrais no topo (SGPR 50,6%, Casa Civil 42,6%)
contra **1,09%** de média em 90 universidades federais — diferença de **33,7×**.
A Casa Civil não pode ser o principal *destino* de pedidos encaminhados. A
segunda hipótese vence, e o campo foi **mantido**.

**Armadilha registrada:** um teste anterior parecia confirmar a sobrescrita (19
de 20 órgãos de taxa alta eram "receptores comprovados"), mas estava
**confundido por volume** — órgãos grandes pertencem ao conjunto de receptores
por serem grandes. A correção de taxa-base mostrou que a pertinência basal já
era de 32,4%. Um teste de sobreposição sem taxa-base é capaz de "confirmar"
qualquer coisa.

### Passo 5 — quantificar, não apenas excluir

Para cada campo suspeito, treinar uma **variante diagnóstica** que o inclui e
medir a diferença. Isso serve a três propósitos: prova que a exclusão importava,
dá o número para publicar, e protege o projeto de reintrodução acidental por
alguém que ache a variável "obviamente útil".

Implementado em `scripts/train.py` como as variantes `with_assunto` e
`LEAKY_with_prazo`. A segunda nunca deve ser implantada, e seu nome diz isso.

### Passo 6 — impor a regra em tempo de execução

Auditoria documentada não impede reintrodução. `lai_triagem/featurize.py` mantém
o conjunto `LEAKAGE_FIELDS` e **recusa a requisição** que traga qualquer um
deles, em vez de pontuar em silêncio:

```
post-hoc field(s) supplied, refusing to score: ['FoiProrrogado'].
```

Falhar ruidosamente é preferível a servir um escore inválido.

---

## 5. Lista final de exclusões e o teste que a decidiu

| Campo | Teste que decidiu | Veredito |
|---|---|---|
| `DataResposta`, `Decisao`, `EspecificacaoDecisao`, `DetalhamentoDecisao`, `MotivoNegativaAcesso`, `PrazoRestricaoAcesso` | Passo 1, documental | *post hoc* por definição |
| `Situacao` | Passo 1 — é o estado corrente, codifica o desfecho | *post hoc* |
| `FoiProrrogado` | Passo 1 — concedida depois da triagem | *post hoc* |
| `AssuntoPedido`, `SubAssuntoPedido`, `Tag` | Passo 2, decaimento por recência | *post hoc* (79,6% ausente a 3 dias) |
| `PrazoAtendimento` e a derivada `prazo_dias` | Passo 3, assinatura estatutária | *post hoc* (+10 d = art. 11 §2) |
| `OrgaoDestinatario` | Passo 4, teste direcional | **legítimo, mantido** |

---

## 6. Lista de verificação transferível

Para qualquer projeto de predição sobre dados abertos administrativos:

1. **Declare o instante de decisão** antes de olhar qualquer coluna.
2. **Não confie na documentação** — verifique o esquema real e a cardinalidade.
3. **Rode o teste de recência no ano corrente.** Anos fechados mentem.
4. **Audite as derivadas, não só os campos brutos.** A aritmética reconstrói o
   que a exclusão removeu.
5. **Compare com a norma escrita.** Prazos legais deixam assinaturas numéricas
   exatas.
6. **Sempre corrija por taxa-base** antes de ler uma sobreposição como evidência.
7. **Quantifique cada vazamento** com uma variante diagnóstica, e nomeie-a de
   modo que ninguém a implante.
8. **Imponha a exclusão em tempo de execução**, não apenas na documentação.
9. **Desconfie da sua melhor variável.** Neste projeto, a de maior ganho era
   vazamento nas duas vezes que olhamos.


---

## Tabela de exclusões, movida do README em 21/09/2026

### Campos excluídos, e por quê

Toda exclusão é **empírica**, não precaucional. Os testes estão em
[`docs/VERIFICATION.md`](VERIFICATION.md). O serviço **recusa** qualquer
requisição que contenha um destes campos.

| Campo excluído | Momento real de preenchimento | Razão da exclusão |
|---|---|---|
| `FoiReencaminhado` | após o encaminhamento | É o próprio alvo |
| `PrazoAtendimento` / `prazo_dias` | reescrito na prorrogação | Mediana 21 d sem prorrogação vs **31 d** com — exatamente os +10 d do art. 11 §2 da LAI. Reimportava `FoiProrrogado`. Detinha **40,5% do ganho** e inflava a PR-AUC em **+22 pp** |
| `FoiProrrogado` | ao conceder a prorrogação | Posterior à triagem |
| `AssuntoPedido` | atribuído **durante** a triagem | **79,6% ausente** em pedidos com 3 dias; 0,000% após respondidos. É saída da triagem, não entrada |
| `SubAssuntoPedido` | idem | 87,5% ausente com 3 dias; ~49% ausente mesmo no longo prazo |
| `Tag` | marcação posterior do SIC | 78,4% ausente; classificação feita depois |
| `Situacao` | estado corrente | Codifica o desfecho |
| `DataResposta`, `Decisao`, `EspecificacaoDecisao`, `DetalhamentoDecisao`, `MotivoNegativaAcesso`, `PrazoRestricaoAcesso` | após a resposta | Posteriores à decisão |
| `ProtocoloPedido` / `protocolo_seq` | atribuído na abertura, mas pela **unidade registradora** | A fatia `[5:11]` do protocolo não é um sequencial neutro: separa **79×** dentro de um mesmo órgão-ano (INSS 2022: 28,38% no 1º quarto contra 0,36% no 4º). Codifica qual unidade registrou, cujo comportamento de encaminhamento é quase determinístico (H5) |
| texto do pedido (`ResumoSolicitacao`, `DetalhamentoSolicitacao`) | na abertura | Disponível, mas **fora de escopo** pelo Termo de Abertura. Exige os arquivos `_Filtrado` (~80 MB/ano contra 7–9 MB) |

Também são removidas da modelagem as **459 linhas** com
`Situacao == "Encaminhada por Outro Órgão"`: estão em trânsito, de modo que seu
`OrgaoDestinatario` é o receptor, não o endereçado.
