# Temporian no lugar da mecânica temporal à mão — 21/09/2026

Auditor: o próprio autor, com o roteiro fixado antes de escrever qualquer
linha. Escopo: substituir a construção de variáveis por **Temporian 0.9.0**,
sem tocar no corte temporal, na métrica, no contrato de serviço nem nas
guardas, e medir se a promessa da biblioteca se confirma.

A pergunta é a de [`../LINHA_BASE_COMPLEXIDADE.md`](../LINHA_BASE_COMPLEXIDADE.md):
o Temporian afirma que uma variável **não consegue** depender do futuro, porque
toda janela é `(t - w, t]` e olhar adiante exige pedir com `leak()`. Se for
verdade, os 14 pontos de invariante mantida à mão caem para perto de zero.

> Os números aqui são os **deste dia**. Para os correntes, sempre
> [`../METRICAS.md`](../METRICAS.md), que é gerado pelo treinamento.

## O veredito, em uma frase

**A promessa se confirma para metade do problema, e a metade que ela resolve é
justamente a que produziu o pior defeito do projeto — mas o preço é alto e
está quase todo fora do código.**

## A regra de aceitação foi cumprida

| Teste 2026 maturado | linha de base | `develop` | este ramo |
|---|---|---|---|
| PR-AUC | 0,1641 | 0,1901 | **0,1901** |
| precisão@5% | 24,80% | 24,23% | **24,23%** |

Não é "dentro do ruído": é igual. E é igual porque foi construído para ser —
antes de trocar a implementação, um arnês comparou as **19 colunas** de
variáveis, linha a linha, contra a implementação original nas 654.718 linhas da
coorte. Divergência em qualquer coluna: zero.

Isso importa mais do que parece. Sem a comparação coluna a coluna, "o número
bateu" seria fraco: duas construções diferentes podem dar o mesmo PR-AUC com
variáveis diferentes, e a igualdade seria coincidência de agregado.

### O controle que veio antes

O Temporian obriga a descer Python, numpy e pandas (ver abaixo). Isso é um
problema de método: se o número mudasse, não haveria como saber se foi a
biblioteca ou a versão. Então o controle foi rodado **primeiro** —
`scripts/train.py` sem uma linha alterada, sob py3.11 + numpy 1.26.4 +
pandas 2.3.3:

| Pilha | PR-AUC | precisão@5% |
|---|---|---|
| py3.12 + numpy 2.5.3 + pandas 3.0.5 (fixada) | 0,1901 | 24,23% |
| py3.12 + pandas 2.3.3 | 0,1901 | 24,23% |
| py3.11 + numpy 1.26.4 + pandas 2.3.3 | 0,1901 | 24,23% |

O recuo de versão não move o resultado. Qualquer diferença medida depois é da
engenharia de variáveis.

## As três medidas

| Medida | `develop` | este ramo | |
|---|---|---|---|
| Linhas de código de variáveis (SLOC do radon) | 141 | **206** | +46% |
| Complexidade média (29 → 34 blocos) | A (4,03) | **A (3,62)** | −10% |
| **Invariantes mantidas à mão** | **14** | **10** | **−29%** |
| precisão@5% no teste maturado | 24,23% | 24,23% | = |
| PR-AUC no teste maturado | 0,1901 | 0,1901 | = |

### Sobre as linhas: o 235 publicado estava medido com duas réguas

A tabela da linha de base soma 235, e esse número não se compara com nada,
porque foi obtido misturando dois critérios: as quatro funções de `train.py`
foram contadas pela **extensão de linhas** (`build_features` = 117 é
exatamente 357 − 241 + 1, comentários e linhas em branco inclusive), e os dois
módulos de `lai_triagem/` pelo **SLOC do radon** (20 e 18 batem com
`uvx radon raw` na casa decimal).

Contar comentário como linha de código pune quem documenta. Este projeto
documenta muito — 79 linhas de comentário só no módulo novo —, então a régua
não é detalhe.

Recontado tudo por SLOC do radon, que é a régua que a própria linha de base já
usava para os dois módulos:

| Onde | `develop` | este ramo |
|---|---|---|
| construção de variáveis | 103 (4 funções em `train.py`) | 168 (`variaveis_temporian.py`) |
| `lai_triagem/codificacao.py` | 20 | 20 |
| `lai_triagem/dados.py` | 18 | 18 |
| **total** | **141** | **206** |

**O Temporian aumentou o código em 46%.** Não há como dourar isso. O aumento
vem de conversão: montar `EventSet`, carregar um `rid` para recuperar a ordem
das linhas, converter de volta para pandas. A lógica temporal encolheu; o
encanamento em volta dela cresceu mais.

### Sobre as invariantes: de 14 para 10, e o 0 era mentira

A primeira medição devolveu **zero**, e zero estava errado. O contador
procurava `searchsorted`, `cumsum`, `merge_asof` — vocabulário do pandas. O
módulo do Temporian não tem nenhum deles e continua tendo defasagem escolhida à
mão, exclusão do próprio evento feita com `- 1` e um relógio intradiário
inventado.

Uma medida que zera quando se troca de biblioteca mede vocabulário, não risco.
`scripts/conta_invariantes.py` passou a relatar duas contagens: a regra
publicada (que reproduz os 14 de `develop`, servindo de aferição) e as
invariantes **declaradas no código** com o comentário `# INVARIANTE:`. São
`grep`-áveis, e estão listadas aqui para que a conferência seja possível:

| # | Onde | O que uma pessoa precisa acertar |
|---|---|---|
| 1 | `taxas_moveis_por_orgao` | defasagem de maturação de 60 dias |
| 2 | `taxas_moveis_por_orgao` | usar data pura, SEM desempate, para o dia da fronteira entrar inteiro |
| 3 | `desfechos_defasados` | a mesma defasagem, de novo |
| 4 | `desfechos_defasados` | `JANELA_TOTAL` fazendo o papel de "desde o começo" |
| 5 | `contagens_do_solicitante` | o relógio intradiário, sem o qual o mesmo dia se vê em bloco |
| 6 | `contagens_do_solicitante` | `- 1` para o pedido não contar a si mesmo |
| 7 | `contagens_do_solicitante` | idem, no par solicitante×órgão |
| 8 | `contagens_do_solicitante` | idem, na contagem de órgãos distintos |
| 9 | `contagens_do_solicitante` | repor o ausente no primeiro pedido de cada cidadão |
| 10 | `build_features` | a ordenação por (`_reg`, `IdPedido`) de que o item 5 depende |

A medida é auto-declarada, e isso é limitação de verdade: quem escreve pode
esquecer de marcar. A defesa é a que o projeto usa em toda parte — isenção
declarada em vez de omissão silenciosa.

## O que ficou melhor

**A classe do H7 deixou de ser alcançável.** É o ganho real, e é grande. O H7
foi o pior defeito de alinhamento do projeto: `merge_asof` reinicia o índice,
`lagged_outcome_sums` devolvia posição onde quem consumia esperava rótulo,
94.145 pedidos ficaram sem histórico e 120.060 receberam o histórico de outra
pessoa. No Temporian o resultado de uma operação móvel **vem preso à amostragem
que o gerou**. Não existe índice intermediário, logo não existe índice
intermediário errado. Não é que ficou mais fácil de acertar: ficou impossível
de errar daquele jeito.

**A fronteira superior da janela virou definição, não convenção.** Sumiram os
dois `searchsorted` com `side="right"` escolhido a dedo, os quatro `cumsum` e a
aritmética `ycum[hi] - ycum[lo]`. Nenhuma operação alcança evento posterior ao
instante amostrado, e alcançar exigiria escrever `leak()` — que é visível na
revisão de um jeito que `side="left"` nunca foi.

**`drop_index` é genuinamente melhor.** Contar órgãos distintos exige ler o
mesmo evento por par e depois por solicitante. No pandas eram dois `groupby` e
um `cumsum` alinhado à mão; aqui é uma operação que reagrupa sem recalcular.

**A complexidade caiu de A(4,03) para A(3,62).** Ressalva registrada na própria
linha de base: as funções de variável já eram todas A, e foi delas que saíram
os três defeitos graves. A medida não protegeu contra nada antes e não protege
agora.

## O que ficou pior

**Custa uma versão inteira do Python.** `temporian==0.9.0` é a última publicada
— 16/04/2024, sem lançamento desde então — e declara `requires_python <3.12`.
Não há roda para cp312, e o pacote tem extensão C++ construída com Bazel, então
compilar do fonte também não sai num ambiente `uv` comum. Foi tentado, falhou.

O projeto exige `>=3.12`. Adotar o Temporian é **descer o projeto inteiro para
3.11**, e isso não fica contido no `pyproject.toml`:

- `scripts/refresh_organ_tables.py` — que não é experimento, é a ferramenta que
  reescreve as tabelas de órgão dentro do artefato de produção — usa f-string
  com aspa reaproveitada (PEP 701, válida só a partir do 3.12). Sob 3.11 são
  **erro de sintaxe**. O arquivo deixou de compilar. Duas linhas corrigidas
  aqui, mas o ponto é que o recuo quebra código que ninguém pensou em olhar.
- O mypy parou de rodar. `uvx --with pandas-stubs mypy` monta ambiente próprio
  em 3.13 com numpy 2, cujos stubs usam `type X = ...` (PEP 695); configurado
  para 3.11, o mypy morre lendo o stub antes de ver uma linha do projeto. Foi
  preciso apontar `python_executable` para o ambiente do projeto e trazer
  `pandas-stubs` para dentro dele, senão a checagem de pandas sumia sem avisar.

**Ele erra em silêncio na pilha atual do projeto.** Este é o achado que mais
incomoda, numa biblioteca vendida como "correção por construção". Temporian
0.9.0 **importa sem reclamar** sob numpy 2 e pandas 3 — e então devolve
conjuntos de eventos **vazios** quando o índice é de texto. Medido: um
`event_set` de 3 eventos indexado por uma coluna de texto vira 0 eventos em
cada grupo, sem exceção, sem aviso. Todo índice deste projeto é texto (órgão,
solicitante). Quem instalasse na pilha fixada e não conferisse teria variáveis
inteiramente zeradas e um modelo plausível.

**`to_pandas` não sabe ler acento.** A conversão de volta faz `astype(str)`
sobre a chave de índice, que é `bytes`, sem informar codificação — ou seja, em
ASCII. Todo órgão do Fala.BR é `SIGLA – Nome`, com travessão U+2013, e a
conversão estoura com `UnicodeDecodeError`. São 877 órgãos, praticamente todos.
Contornado descartando o índice antes de converter, mas é um defeito da
biblioteca em dado brasileiro comum.

**Ficou 12 vezes mais lento.** A construção de variáveis passou de **3,4 s**
para **41,6 s** nas mesmas 654.718 linhas. Não inviabiliza um treinamento que
roda em minutos; inviabilizaria iteração interativa.

## O que o Temporian NÃO resolveu

Esta seção é a que importa para a decisão, porque é o que se esperava e não
veio.

**1. O evento vê a si mesmo, e o mesmo dia é simultâneo.** Em `(t - w, t]`, o
`t` está dentro. Dois eventos no mesmo instante somam um ao outro *e a si
próprios* — verificado na sonda. `DataRegistro` é data sem hora, então **todo
pedido do mesmo dia é simultâneo**: é exatamente o vazamento de mesmo dia que a
auditoria externa mediu em 159.320 linhas, com 22.793 rótulos positivos. O
Temporian não impede nada disso. Ele o reproduz fielmente, porque para ele um
evento em `t` já aconteceu em `t` — o que é uma posição defensável, e é
irrelevante para quem precisa da garantia.

A consequência prática: foi preciso **inventar um relógio**, somando um
milésimo de segundo por posição na ordem de `IdPedido`, para que os pedidos do
dia parassem de se enxergar. Essa construção não é verificada por nada. Se o
passo fosse grande demais e cruzasse a meia-noite, ou se a ordenação de origem
mudasse, as contagens mudariam em silêncio.

**2. A maturação continua inteiramente por conta de quem modela.** Nada na
biblioteca sabe que `FoiReencaminhado` leva 60 dias para se firmar. A defasagem
entra à mão, em dois lugares, como `- lag_dias` no instante de amostragem.
Esquecer em um dos dois deixaria metade do histórico maduro e metade não, sem
erro visível. É o defeito Fix 2, e ele continua igualmente possível.

**3. A exclusão do próprio evento é três `- 1` à mão.** Trocou-se `cumcount`
por `moving_count` seguido de subtração. Em número de pontos frágeis, empate.

**4. O H8 e o H10 estão fora do alcance.** O H8 (prior de suavização calculado
com rótulo de 2025 e 2026) e o H10 (codificação por órgão sem cross-fitting)
não são problemas de janela temporal — são de **qual partição** alimenta uma
estatística. O Temporian não tem opinião sobre partição. `codificacao.py`
ficou intocado, com as mesmas 0 invariantes de token e o mesmo risco.

Vale reparar: dos três defeitos graves que a linha de base atribui a esses 14
pontos, o Temporian elimina **um** (H7). O H8 e o H10 sobrevivem inteiros.

## Recomendação

**Não adotar como está.** O ganho é real e é específico: a classe do H7 morre.
Mas o preço é descer o Python do projeto inteiro para uma versão que já quebrou
uma ferramenta de produção, depender de um pacote sem lançamento há dezessete
meses, e conviver com uma biblioteca que devolve resultado vazio em silêncio se
alguém atualizar numpy.

Se o objetivo é matar a classe do H7, o caminho barato é outro: proibir
`merge_asof` na construção de variáveis e exigir que toda agregação defasada
volte alinhada por posição — que é o que o módulo novo faz, e que dá para fazer
em pandas. O ganho verificável do Temporian é de **quatro invariantes em
catorze**, e nenhuma delas é o H8 nem o H10.

## Como reproduzir

```bash
uv sync
uv run python scripts/train.py
uv run python scripts/conta_invariantes.py
```
