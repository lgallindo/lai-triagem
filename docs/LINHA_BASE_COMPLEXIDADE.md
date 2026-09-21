# Linha de base de complexidade — medida em 21/09/2026

Este arquivo existe por um motivo só: daqui a pouco duas versões alternativas
da engenharia de variáveis vão ser escritas, uma com **Temporian** e outra com
**Featuretools**. Para dizer depois se alguma delas ficou mais simples, é
preciso ter medido o estado atual **antes** de começar. Senão a resposta vira
questão de gosto.

Estado medido: commit `7ef46bb`, ramo `main`.

## As três medidas

### 1. Quantas linhas de código constroem as variáveis

| Onde | Linhas |
|---|---|
| `build_features` (em `scripts/train.py`) | 117 |
| `organ_rolling` | 33 |
| `lagged_outcome_sums` | 31 |
| `organ_birth_table` | 16 |
| `lai_triagem/codificacao.py` | 20 |
| `lai_triagem/dados.py` | 18 |
| **total** | **235** |

Contagem de linhas efetivas de código, sem comentário nem docstring.

### 2. Quão emaranhado está esse código

Complexidade ciclomática, pela ferramenta `radon`. Em termos simples: quantos
caminhos diferentes a execução pode tomar dentro de uma função. Mais caminhos,
mais difícil de conferir e mais lugares onde um erro se esconde.

| Função | Nota |
|---|---|
| `main` | **C (17)** — a mais emaranhada |
| `organ_birth_table` | B (7) |
| `fit_calibration` | B (6) |
| `build_features` | A (5) |
| `organ_rolling` | A (4) |
| `lagged_outcome_sums` | A (1) |
| média das 29 funções | **A (4,03)** |

`main` ser a pior não surpreende: ela orquestra tudo. As funções de variável
estão todas em A, o que é bom — e ainda assim foi delas que saíram os três
defeitos graves. **Complexidade baixa não protegeu contra nada.** Vale
registrar isso, porque é um argumento contra usar só esta medida.

### 3. Invariantes garantidas à mão — a medida que importa

**14 ocorrências** de mecânica temporal manual: `searchsorted`, `shift`,
`cumsum`, `merge_asof`, escolha explícita de `side="left"` contra
`side="right"`, e a preservação de rótulo antes de um `merge_asof`.

Cada uma dessas é um lugar onde **uma pessoa tem de acertar na mão** a regra
"não olhe para o futuro". Não há verificação automática; o código roda igual se
estiver errado.

E errou. Dos defeitos graves encontrados neste projeto:

- **H7** — `merge_asof` devolvia posição onde o código esperava rótulo de
  linha. 94.145 pedidos ficaram sem histórico e 120.060 receberam o histórico
  de outra pessoa.
- **H8** — o prior de suavização foi calculado com dados de 2025 e 2026, ou
  seja, com o futuro.
- **H10** — a taxa por órgão era calculada com o próprio pedido incluído.

Os três são falhas de invariante mantida à mão, nos 14 pontos acima.

**É esta a medida que decide a comparação.** O Temporian afirma impedir esta
classe de erro por construção: nele, uma variável não *consegue* depender do
futuro, a menos que se peça explicitamente com `tp.leak()`. Se aquelas 14
ocorrências caírem para perto de zero, o ganho é real, mesmo que o número de
linhas suba.

## Como comparar, depois

Rodar as mesmas três medidas em cada ramo e preencher. Esta cópia está no ramo
`experimento/featuretools`, então preenche a coluna dele; a do Temporian é
preenchida na cópia do outro ramo.

| Medida | `main` (hoje) | Temporian | Featuretools |
|---|---|---|---|
| Linhas de código de variáveis | 235 | | *ver abaixo* |
| Complexidade média | A (4,03) | | A (3,79) |
| **Invariantes à mão** | **14** | | **11** |
| precisão@5% no teste maturado | 24,23% | | 24,23% |
| PR-AUC no teste maturado | 0,1901 | | 0,1901 |
| tempo de construção das variáveis | 3,4 s | | **3.325 s** |

A última linha não estava prevista, e entrou porque sem ela a comparação
mentiria por omissão: 3,4 s contra 55 minutos é uma diferença de natureza, não
de grau. Detalhe em [`auditorias/`](auditorias/INDICE.md).

### Correção de método: o 235 foi medido com duas réguas

Achado ao preencher a coluna, e registrado aqui porque invalida comparação,
não só estética. As quatro funções de `train.py` foram contadas pela
**extensão de linhas** — `build_features` = 117 é exatamente 357 − 241 + 1,
comentário e linha em branco inclusive —, enquanto `codificacao.py` (20) e
`dados.py` (18) vieram do **SLOC do `radon`**, que descarta comentário e
docstring. Os dois últimos batem com `uvx radon raw` na casa decimal; os
quatro primeiros, não.

Somar as duas réguas dá 235, e 235 não se compara com nada. Pior: contar
comentário como linha de código pune quem documenta, o que neste projeto é
incentivo invertido.

Recontado tudo por SLOC do `radon`, que é a régua que a própria tabela já
usava para metade das linhas:

| Medida | `main` | Temporian | Featuretools |
|---|---|---|---|
| construção de variáveis | 103 | | 180 |
| `lai_triagem/codificacao.py` | 20 | | 20 |
| `lai_triagem/dados.py` | 18 | | 18 |
| **total, régua única** | **141** | | **218** |

Comandos, para que não haja uma terceira régua:

```bash
uvx radon raw lai_triagem/codificacao.py
uvx radon cc scripts/train.py lai_triagem/ -a
uv run python scripts/conta_invariantes.py
```

O escopo do `radon cc` acima é o que reproduz os 29 blocos e o A (4,03)
publicados — foi conferido antes de medir qualquer ramo.

### A contagem de invariantes virou script

Estava sendo feita a olho, e medida contada a olho não se compara entre ramos.
`scripts/conta_invariantes.py` aplica a regra desta página de forma executável
e devolve **14** em `develop`, o que serve de aferição: a régua reproduz a
contagem à mão que substitui.

Ela relata **duas** contagens, e a segunda existe por um motivo concreto. A
regra publicada só conhece o vocabulário do pandas, e uma implementação que
troque de vocabulário pode zerá-la sem ter removido risco nenhum. A segunda
contagem são as invariantes declaradas no código com o comentário
`# INVARIANTE:`, que é `grep`-ável e auditável uma a uma.

Os dois últimos vêm de [`METRICAS.md`](METRICAS.md), que é gerado pelo
treinamento.

**Regra da comparação:** o ramo tem de reproduzir o resultado atual dentro do
ruído. Uma versão mais simples que piora o modelo não é uma versão mais
simples — é uma versão diferente. E se algum ramo produzir número *melhor*, a
primeira suspeita deve ser vazamento novo, não ganho: foi o que aconteceu todas
as vezes neste projeto.
