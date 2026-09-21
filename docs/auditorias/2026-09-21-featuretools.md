# Featuretools no lugar da mecânica temporal à mão — 21/09/2026

Auditor: o próprio autor, com o roteiro fixado antes de escrever qualquer
linha. Escopo: substituir a construção de variáveis por **Featuretools
1.31.0**, sem tocar no corte temporal, na métrica, no contrato de serviço nem
nas guardas, e medir se a promessa da biblioteca se confirma.

A pergunta é a de [`../LINHA_BASE_COMPLEXIDADE.md`](../LINHA_BASE_COMPLEXIDADE.md):
o `cutoff_time` declara, por linha, o instante até o qual se pode olhar. Se a
promessa for sólida, os 14 pontos de invariante mantida à mão caem para perto
de zero.

> Os números aqui são os **deste dia**. Para os correntes, sempre
> [`../METRICAS.md`](../METRICAS.md), que é gerado pelo treinamento.

## O veredito, em uma frase

**A biblioteca acerta a fronteira entre dias e não tem modelo nenhum para a
ordem dentro do dia — que é onde os dados deste projeto vivem — e o preço
disso é um custo por corte que torna a construção mil vezes mais lenta.**

## A regra de aceitação foi cumprida

| Teste 2026 maturado | linha de base | `develop` | este ramo |
|---|---|---|---|
| PR-AUC | 0,1641 | 0,1901 | **0,1901** |
| precisão@5% | 24,80% | 24,23% | **24,23%** |

Não é "dentro do ruído": é igual.

Antes de trocar a implementação, um arnês comparou as **12 colunas** de
variáveis derivadas, linha a linha, contra a implementação original numa
amostra de 15.000 solicitantes (35.067 pedidos, 663 órgãos, com colisões de
mesmo dia em abundância). Divergência em qualquer coluna: zero. A comparação
nas 654.718 linhas foi abandonada por custo — duas passagens completas de
construção de variáveis levariam mais de duas horas —, e o treinamento
cumpre o papel: qualquer divergência de variável moveria o PR-AUC.

### O controle que veio antes

O Featuretools obriga a descer o pandas (ver abaixo). Se o número mudasse, não
haveria como separar o efeito da biblioteca do efeito da versão. Então o
controle foi rodado **primeiro** — `scripts/train.py` sem uma linha alterada:

| Pilha | PR-AUC | precisão@5% |
|---|---|---|
| py3.12 + numpy 2.5.3 + pandas 3.0.5 (fixada) | 0,1901 | 24,23% |
| py3.12 + pandas 2.3.3 | 0,1901 | 24,23% |

O recuo do pandas não move o resultado.

## As três medidas

| Medida | `develop` | este ramo | |
|---|---|---|---|
| Linhas de código de variáveis (SLOC do radon) | 141 | **218** | +55% |
| Complexidade média (29 → 33 blocos) | A (4,03) | **A (3,79)** | −6% |
| **Invariantes mantidas à mão** | **14** | **11** | **−21%** |
| precisão@5% no teste maturado | 24,23% | 24,23% | = |
| PR-AUC no teste maturado | 0,1901 | 0,1901 | = |
| tempo de construção das variáveis | 3,4 s | **3.325 s** | **978×** |

Sobre a régua de linhas, ver a correção de método registrada em
[`../LINHA_BASE_COMPLEXIDADE.md`](../LINHA_BASE_COMPLEXIDADE.md): o total de
235 publicado misturava dois critérios, e a coluna acima usa um só.

### As 11 invariantes, uma a uma

`scripts/conta_invariantes.py` relata 3 por token e 8 declaradas. As três por
token são os `cumcount`/`cumsum` do pandas que sobraram — e sobraram porque o
Featuretools não alcança a ordem intradiária. Duas delas estão sob uma única
marca `# INVARIANTE:`, contadas à parte pela mesma convenção por ocorrência que
a linha de base usou ao contar `searchsorted` e `side=` na mesma linha.

| # | Onde | O que uma pessoa precisa acertar |
|---|---|---|
| 1 | `taxas_moveis_por_orgao` | defasagem de maturação de 60 dias |
| 2 | `desfechos_defasados` | a mesma defasagem, num segundo lugar |
| 3 | `contagens_do_solicitante` | o corte ser a véspera, e não o dia do pedido |
| 4–6 | `contagens_do_solicitante` | os três `cumcount`/`cumsum` intradiários |
| 7 | `contagens_do_solicitante` | somar os dois pedaços sem mudar convenção |
| 8 | `contagens_do_solicitante` | órgãos estreados hoje, antes desta linha |
| 9 | `contagens_do_solicitante` | dias desde o último pedido, com o ramo do mesmo dia |
| 10 | `build_features` | a ordenação por (`_reg`, `IdPedido`) de que 4–9 dependem |
| 11 | `_agrega` | a junção de volta, exigida pela proibição de corte duplicado |

## O que ficou melhor

**As janelas por órgão ficaram genuinamente declarativas.** `organ_rolling`
tinha dois `searchsorted` com `side="right"` escolhido a dedo, um `cumsum` e a
aritmética `ycum[hi] - ycum[lo]`. Virou `cutoff_time` mais
`training_window="90 days"`. A fronteira `(corte - w, corte]` foi conferida na
sonda e é exatamente a que o original mantinha por convenção — aqui ela é
contratada, não lembrada.

**O `merge_asof` e os `cumsum` dos desfechos sumiram.** A agregação até
`t - 60d` é um argumento, não um algoritmo.

**O modelo de entidades documenta a intenção.** Declarar que pedidos pertencem
a um solicitante, a um órgão e a um par torna explícito o que estava implícito
em chaves de `groupby` espalhadas. Um leitor novo entende o desenho mais rápido.

## O que ficou pior

**Custa o pandas do projeto.** O Featuretools guarda o esquema das tabelas num
acessor do pandas (`df.ww`, do `woodwork`). Em **pandas 3.0.5 esse estado não
sobrevive**: `df.ww.init(...)` roda sem erro e, na linha seguinte,
`df.ww.schema` é `None`; a primeira chamada de `add_dataframe` morre com
`WoodworkNotInitError`. Não há configuração que contorne. Adotar o
Featuretools é ficar em pandas 2.x até o `woodwork` ser portado.

Junto vem `setuptools<81`: o `woodwork` importa `pkg_resources` no topo do
módulo, e o setuptools 81 removeu esse pacote. A biblioteca depende de uma API
que a própria comunidade aposentou.

**É mil vezes mais lento, e a causa é estrutural.** Na coorte inteira, a
construção de variáveis passou de **3,4 s** para **3.325 s** — de três
segundos para 55 minutos, 978×. Um treinamento completo foi de 64 s para
3.354 s. Na amostra de 35.067 linhas a razão é a mesma: 0,2 s contra 212 s.

Não é constante de inicialização. O `cutoff_time` é avaliado **por corte
distinto**, e para cada um a biblioteca refiltra e reagrega a tabela de
eventos inteira. São ~1.700 datas distintas contra 654.718 eventos, e o
produto é o custo.

Medição isolada do mecanismo: **6,3 ms por corte distinto**. Um corte por
linha — que é o que a ordem por `IdPedido` exigiria — daria mais de uma hora
por passagem, e são seis.

**`n_jobs` não funciona sem Dask.** `n_jobs=-1` levanta `ImportError` pedindo
Dask. A saída óbvia para o problema de custo exige outra dependência pesada.

**7,6 MB de avisos por execução.** Dois `FutureWarning` do `woodwork`, emitidos
a cada corte, mais o `pkg_resources`. Foi preciso silenciá-los por módulo para
que a saída do treinamento continuasse legível.

## O que o Featuretools NÃO resolveu

**1. O corte inclui o próprio instante — e portanto o próprio rótulo.** Com
`cutoff_time = t`, o `SUM(eventos.y)` da linha em `t` soma o `y` dela mesma.
Verificado na sonda: `COUNT` e `SUM` deram valores idênticos, porque cada
evento entrava na própria agregação. A configuração mais óbvia da biblioteca —
"o corte é o instante da linha" — é vazamento de alvo direto.

E o reflexo de corrigir recuando o corte **não funciona**: com
`cutoff_time = t - ε` e o pedido no papel de alvo, o Featuretools considera que
a linha ainda não existe e devolve zeros com o próprio `y` em `NaN`, para
*todas* as linhas. Foi preciso reformular o modelo inteiro, pondo a tabela-pai
como alvo.

**2. A ordem dentro do dia não tem representação.** `DataRegistro` é data sem
hora, e a ordem real é `IdPedido`, outra coluna. O Featuretools só conhece um
instante. Como o corte é `<=`, um corte na data do pedido inclui o pedido e
todos os irmãos do mesmo dia — o vazamento de mesmo dia que a auditoria
externa mediu em 159.320 linhas, com 22.793 rótulos positivos.

A saída foi partir a conta em dois: o Featuretools responde "quantos até a
véspera" e o pandas responde "e mais estes, hoje, antes de mim", com
`cumcount`. **Metade das contagens voltou para o pandas**, com três tokens de
acumulação ordenada e a dependência da ordenação de origem. A biblioteca não
cobre o caso, e o caso não é exótico: é o formato em que a CGU publica.

**3. Corte duplicado é erro, e a junção de volta ressuscita o H7.**
`(instância, tempo)` repetido levanta `AssertionError` — e milhares de pedidos
chegam ao mesmo órgão no mesmo dia, então a repetição é a regra. É obrigatório
deduplicar, calcular, e **recasar o resultado com as linhas**. Essa junção é a
mesma operação cujo erro produziu o H7: 94.145 pedidos sem histórico e 120.060
com o histórico de outra pessoa.

Ficou mais segura, e é justo dizer por quê: com `cutoff_time_in_index=True` a
junção é por `(chave, corte)` explícito, `validate="many_to_one"` recusa
cardinalidade errada, e um `assert` confere a contagem de linhas. Mas ela
**existe**, foi escrita à mão, e é a invariante nº 11.

**4. `training_window` devolve zero em silêncio sem `add_last_time_indexes()`.**
Este é o achado que mais incomoda. A mesma consulta que devolve 1, 2 e 3 com o
índice de último tempo configurado devolve **0, 0 e 0** sem ele. Não é erro,
não é aviso: é zero. E `add_last_time_indexes()` não é chamado por padrão nem
exigido pela API. Numa biblioteca cuja proposta é cuidar do tempo por você, a
janela temporal é precisamente o que falha calado quando falta uma linha de
configuração.

**5. O H8 e o H10 estão fora do alcance.** O H8 (prior de suavização calculado
com rótulo de 2025 e 2026) e o H10 (codificação por órgão sem cross-fitting)
não são problemas de janela — são de **qual partição** alimenta uma
estatística. O `cutoff_time` não tem opinião sobre partição.
`codificacao.py` ficou intocado, com o mesmo risco.

Dos três defeitos graves que a linha de base atribui aos 14 pontos, o
Featuretools deixa **zero** estruturalmente impossíveis: o H7 continua
possível, apenas mudou de junção; o H8 e o H10 nem são endereçados.

## Recomendação

**Não adotar.** É a recomendação mais firme das duas. O Temporian ao menos mata
uma classe de defeito; o Featuretools não mata nenhuma, e cobra caro.

O caso deste projeto é o pior possível para o desenho da biblioteca: carimbos
com granularidade de dia, ordem verdadeira numa coluna separada, e volume alto
com poucas datas distintas. O `cutoff_time` foi desenhado para "uma predição
por cliente por mês", não para "uma predição por evento, com a ordem dos
eventos importando dentro do dia".

O que vale levar embora, e não precisa da dependência: **declarar a janela em
vez de calculá-la**. `training_window="90 days"` é mais legível que dois
`searchsorted` com `side=` escolhido a dedo, e a legibilidade é real. Dá para
ter isso com uma função de dez linhas em pandas, com a fronteira testada — que
é o que já existe, sem o custo.

## Como reproduzir

```bash
uv sync
uv run python scripts/train.py
uv run python scripts/conta_invariantes.py
```

Aviso de tempo: a construção de variáveis leva mais de uma hora nas 654.718
linhas. Não é defeito da configuração; é o custo do `cutoff_time`.
