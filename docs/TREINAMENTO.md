# Procedimento de treinamento — reproduzível

Este documento descreve exatamente o que `scripts/train.py` faz, com que
configuração de LightGBM e em que ambiente. Qualquer divergência entre este
texto e o script é um defeito: o script é a fonte da verdade.

## 1. Ambiente exato

Capturado no ambiente que produziu os artefatos versionados:

| Componente | Versão |
|---|---|
| Python | 3.12.3 |
| **LightGBM** | **4.7.0** |
| pandas | 3.0.5 |
| NumPy | 2.5.3 |
| scikit-learn | 1.9.1 (apenas `roc_auc_score`, `average_precision_score`) |
| PyArrow | 25.0.1 |
| BentoML | 1.4.39 (só para servir) |
| pydantic | 2.13.5 (só para servir) |

Reproduzir com:

```bash
uv sync                # treino
uv sync --extra serve  # treino + serviço
```

As versões estão **fixadas** em `pyproject.toml`. Trocar qualquer uma invalida a
reprodução bit-a-bit.

### Hardware de referência

| Item | Valor |
|---|---|
| CPU | 13th Gen Intel(R) Core(TM) i7-13700 |
| Núcleos visíveis no WSL | 6 |
| RAM visível no WSL | 19 GiB |
| GPU | nenhuma utilizada — LightGBM em CPU |

O host tem 24 processadores lógicos, mas o WSL expõe 6. Por isso
`num_threads=6`. **Alterar `num_threads` altera os resultados**: a ordem de
redução em ponto flutuante do LightGBM depende do número de threads, então o
determinismo exato exige o mesmo valor.

## 2. Procedência dos dados

| Item | Valor |
|---|---|
| Fonte | <https://dadosabertos-download.cgu.gov.br/FalaBR/Arquivos_FalaBR/> |
| Arquivos | `Pedidos_csv_{2022,2023,2024,2025,2026}.zip` |
| Retrato | **20260914** (prefixo dos CSVs internos) |
| Conteúdo de cada zip | `<retrato>_Pedidos_csv_<ano>.csv` e `<retrato>_SolicitantesPedidos_csv_<ano>.csv` |
| Codificação | **UTF-16**, separador `;` |
| Volume | ~38 MB comprimidos; ~450 MB expandidos |

Os arquivos `_Filtrado` (com texto do pedido, ~80 MB/ano) **não** são usados:
texto está fora de escopo pelo Termo de Abertura.

O retrato é parte da identidade do experimento. A CGU republica os mesmos URLs
com dados atualizados, portanto **baixar hoje não reproduz estes números** — o
prefixo do CSV interno revela o retrato efetivamente obtido.

## 3. Construção da coorte

```
655.177 linhas brutas (2022–2026)
  − 459 linhas com Situacao == "Encaminhada por Outro Órgão"
= 654.718 linhas modeladas
```

A exclusão é necessária porque essas linhas estão em trânsito: o
`OrgaoDestinatario` delas é o órgão **receptor**, não o endereçado (ver H2 em
`VERIFICATION.md`).

Junção: `Pedidos LEFT JOIN Solicitantes ON IdSolicitante`, com
`drop_duplicates("IdSolicitante")` no lado direito. As colunas `UF` e
`Municipio` do lado dos solicitantes são renomeadas para `UF_sol` e
`Municipio_sol` para evitar colisão.

## 4. Corte temporal

Nunca aleatório: o modelo pontua chegadas futuras.

| Partição | Anos | n | Taxa de positivos |
|---|---|---|---|
| Treino | 2022, 2023, 2024 | 391.425 | 8,03% |
| Validação | 2025 | 150.142 | 6,77% |
| Teste | 2026 | 113.151 | 5,23% |
| Teste maturado | 2026, registro ≤ 2026-07-16 | 86.096 | 5,75% |

`MATURITY_DAYS = 60`. A partição maturada existe porque o retrato é de
2026-09-14: um pedido de setembro teve dias, não meses, para ser reencaminhado.
Verificado: 5,75% de positivos entre maturados contra **3,60%** entre recentes —
censura à direita confirmada.

## 5. Engenharia de variáveis

14 categóricas e 6 numéricas, 20 no total. Todas derivam apenas do que existe
na chegada.

### Categóricas (nativas no LightGBM)

`Esfera`, `UF`, `Municipio`, `OrgaoDestinatario`, `FormaResposta`,
`OrigemSolicitacao`, `TipoDemandante`, `Genero`, `Escolaridade`, `Profissao`,
`TipoPessoaJuridica`, `Pais`, `UF_sol`, `Municipio_sol`.

Codificação: inteiros ordenados alfabeticamente por nível, aprendidos **só no
treino**. Nível ausente ou inédito recebe `-1`, que o LightGBM trata como
faltante (daí o aviso `Met negative value in categorical features`, esperado).

O mapa completo vai para `artifacts/preprocessor.json`, de modo que a
inferência reproduz a codificação sem `pickle`.

### Numéricas

| Variável | Fórmula |
|---|---|
| `orgao_rate` | ver 5.1 |
| `idade` | `(DataRegistro − DataNascimento) / 365,25`, arredondada a 1 decimal; `NaN` fora de [10, 110] |
| `reg_month`, `reg_dow`, `reg_day` | mês, dia da semana, dia do mês de `DataRegistro` |
| `uf_match` | `1` se `UF_sol == UF` |

`DataRegistro` é **somente data** nos dados reais, apesar de a documentação da
CGU indicar `DD/MM/AAAA HH:MM:SS`. Não há variável de hora do dia.

### 5.1 Codificação de alvo por órgão (`orgao_rate`)

```
orgao_rate(o) = (positivos(o) + 50 × taxa_base) / (n(o) + 50)
```

com `taxa_base = 0,080268` (média do treino) e prior de 50 pedidos. **Ajustada
exclusivamente nos anos de treino** e aplicada inalterada a validação e teste —
é isso que impede vazamento temporal. Órgão não visto recai na taxa-base.

A tabela resultante é embarcada em `artifacts/preprocessor.json`, portanto o
serviço é sem estado.

## 6. Configuração do LightGBM

Literal, como em `scripts/train.py`:

```python
params = dict(
    objective="binary",
    metric="average_precision",
    learning_rate=0.1,
    num_leaves=31,
    max_bin=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_data_in_leaf=100,
    num_threads=6,
    verbose=-1,
    seed=42,
)
booster = lgb.train(
    params, dtr,
    num_boost_round=600,
    valid_sets=[dva],
    callbacks=[lgb.early_stopping(40, verbose=False)],
)
```

Justificativa de cada escolha, dado que **velocidade de treinamento tem
precedência sobre acurácia** neste projeto:

| Parâmetro | Valor | Por quê |
|---|---|---|
| `objective` | `binary` | Alvo binário `FoiReencaminhado == "Sim"` |
| `metric` | `average_precision` | O produto é uma fila limitada por capacidade; PR-AUC reflete isso, ROC-AUC não. A parada antecipada otimiza a métrica que importa |
| `learning_rate` | 0,1 | Padrão; sem busca de hiperparâmetros por decisão de projeto |
| `num_leaves` | 31 | Padrão; suficiente dada a baixa dimensionalidade |
| `max_bin` | **63** | Metade do padrão (255). Principal alavanca de velocidade; custo de acurácia desprezível aqui |
| `feature_fraction` | 0,8 | Regularização leve |
| `bagging_fraction` / `bagging_freq` | 0,8 / 1 | Regularização leve, acelera cada iteração |
| `min_data_in_leaf` | 100 | Evita folhas ditadas por órgãos raros |
| `num_threads` | 6 | Núcleos visíveis no WSL; **afeta o determinismo** |
| `seed` | 42 | Fixa amostragem de bagging e de variáveis |
| `num_boost_round` | 600 | Teto; a parada antecipada decide de fato |
| `early_stopping` | 40 | Avaliada em 2025, nunca em 2026 |

**Sem balanceamento de classes**: nem `scale_pos_weight` nem `is_unbalance`. O
produto ordena, não classifica em 0,5, então distorcer as probabilidades não
traria benefício.

### Modelo resultante (variante de produção)

| Item | Valor |
|---|---|
| Arquivo | `artifacts/model_arrival.txt` |
| Formato | texto nativo do LightGBM, `version=v4` |
| Árvores | **161** (parada antecipada; teto era 600) |
| `num_leaves` | 31 |
| `num_class` | 1, `objective=binary sigmoid:1` |
| Tamanho | ~1,0 MB — versionável em git puro |
| Tempo de ajuste | **1,93 s** |

## 7. As três variantes treinadas

Todas com o mesmo corte e os mesmos hiperparâmetros.

| Variante | Variáveis | Papel |
|---|---|---|
| `arrival` | 14 cat + 6 num | **Produção.** Só o que existe na chegada |
| `with_assunto` | + `AssuntoPedido`, `SubAssuntoPedido` | Diagnóstica: quantifica o vazamento H1 |
| `LEAKY_with_prazo` | + `prazo_dias` | Diagnóstica: quantifica o vazamento H3. **Nunca publicar** |

Custo medido dos vazamentos (PR-AUC):

| Partição | `arrival` | `with_assunto` | `LEAKY_with_prazo` |
|---|---|---|---|
| validação 2025 | 0,2053 | 0,2183 (+1,31 pp) | 0,4318 (**+22,65 pp**) |
| teste 2026 | 0,1602 | 0,1549 (−0,53 pp) | 0,3772 (**+21,70 pp**) |
| teste maturado | 0,1723 | — | 0,3965 (**+22,42 pp**) |

`AssuntoPedido` **piora** o teste fora do tempo. `prazo_dias` quase dobra o
desempenho aparente. Ambos são inviáveis em produção.

## 8. Protocolo de avaliação

- **Métrica primária: precisão@k**, não AUC. O entregável é uma fila limitada
  pela capacidade de analistas seniores. Reportada em k ∈ {1%, 2%, 5%, 10%, 20%}.
- **O ganho (`lift`) usa a taxa-base da própria partição.** Defeito corrigido: a
  versão anterior passava a taxa-base do treino (8,03%) para o teste (5,23%),
  subestimando todo ganho de teste por 1,53×.
- **Linha de base obrigatória**: ordenar por `orgao_rate` puro, sem modelo. Como
  76% do ganho do modelo é identidade do órgão, essa é a barra real.
- **Auditoria de equidade** na fila de 10%, por `Escolaridade`, `Genero` e
  `TipoDemandante`, comparando participação na fila com participação na
  população.

## 9. Reprodução

```bash
git clone git@github.com:lgallindo/lai-triagem.git && cd lai-triagem
uv sync
mkdir -p data/raw data/interim artifacts
for y in 2022 2023 2024 2025 2026; do
  curl -sS -o data/raw/Pedidos_csv_$y.zip \
    "https://dadosabertos-download.cgu.gov.br/FalaBR/Arquivos_FalaBR/Pedidos_csv_$y.zip"
  unzip -oq data/raw/Pedidos_csv_$y.zip -d data/interim/
done
uv run python scripts/train.py
```

Tempo total: **~20 s** (4,8 s de carga e featurização; ~2 s por ajuste).

Auditorias de vazamento, independentes do treinamento:

```bash
uv run python scripts/verify_leakage_v2.py   # H1, H2 (duplicatas, Recursos)
uv run python scripts/verify_h2_final.py     # H2 direcional
uv run python scripts/verify_h2_baserate.py  # H2 correção de taxa-base
uv run python scripts/verify_h3_prazo.py     # H3 prazo, H4 demografia
```

## 10. Limites do determinismo

Reproduz exatamente com: mesmo retrato de dados, mesmas versões fixadas,
`num_threads=6`, `seed=42`.

Não reproduz exatamente se: o retrato da CGU mudou (mais provável — eles
republicam os mesmos URLs), `num_threads` difere, ou a versão do LightGBM
difere. As conclusões qualitativas — vazamento de `prazo_dias`, empate com a
linha de base, ausência das variáveis demográficas — são robustas a essas
variações; os decimais não.
