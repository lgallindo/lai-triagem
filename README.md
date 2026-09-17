# lai-triagem — risco de reencaminhamento de pedidos LAI (Fala.BR)

Sinaliza cada pedido de acesso à informação que chega à plataforma federal
**Fala.BR** como `ALTO RISCO` ou `BAIXO RISCO` de reencaminhamento interno
(campo `FoiReencaminhado`), para que analistas seniores do SIC sejam dirigidos
aos pedidos com maior chance de roteamento incorreto.

> ## Leia antes de usar o escore
>
> Cinco famílias de variáveis foram **excluídas por vazamento**, incluindo as
> duas de maior ganho aparente: `prazo_dias` (+22 pp de PR-AUC) e `protocolo_seq`
> (+26 pp). O serviço mantém `/score_baseline` — a consulta por órgão, sem
> modelo — como comparação permanente, porque houve uma fase do projeto em que
> ela empatava com o modelo. Hoje não empata mais, e a auditoria completa está em
> [`docs/VERIFICATION.md`](docs/VERIFICATION.md) e
> [`docs/CAMPOS_POST_HOC.md`](docs/CAMPOS_POST_HOC.md).

## Resultado principal

Conjunto de teste 2026 maturado (registrados com ≥60 dias de antecedência do
retrato), treino em 2022–2024, validação em 2025. Taxa-base 5,75%.
Modelo de produção: **30 variáveis, 98 árvores, 1,7 s de ajuste**.

| Escore | ROC-AUC | PR-AUC | prec@1% | prec@5% | prec@10% |
|---|---|---|---|---|---|
| Consulta histórica por órgão (sem modelo) | 0,7434 | 0,1641 | 36,12% | 24,79% | 17,79% |
| **LightGBM, 98 árvores** | **0,8001** | **0,2556** | **52,03%** | **30,99%** | **23,28%** |
| Ganho relativo | — | **+56%** | **+44%** | **+25%** | **+31%** |

**Ganho de 5,4× na fila dos 5% mais arriscados.** O modelo supera com folga a
consulta simples — o que **não era verdade** nas versões anteriores deste
repositório, e passou a ser depois de duas descobertas:

- **A experiência do solicitante é específica do órgão.**
  `n_pedidos_previos_neste_orgao` vale 6,66% do ganho contra 1,08% do agregado
  `n_pedidos_previos`. Estreantes são reencaminhados a 8,33%, veteranos com 50+
  pedidos a 5,35%: quem já usou a LAI aprende qual órgão endereçar.
- **A taxa por órgão precisa ser móvel.** `orgao_rate_movel_90d` é a segunda
  variável mais importante (16,13%). A tabela estática ajustada em 2022–2024
  estava desatualizada — a taxa-base caiu de 8,03% para 5,23% no período.

O efeito é visível numa única requisição. Mesmo pedido, mesmo órgão:

```
estreante (0 pedidos)            P=0,2965  ALTO RISCO
veterano (80 pedidos, 12 aqui)   P=0,0914  BAIXO RISCO
```

## Contrato de dados: o que o chamador informa

As sete variáveis de histórico do solicitante são **dado pessoal** e por decisão
explícita **não são embarcadas no artefato** — o chamador as informa, e o
Fala.BR já as possui. O artefato contém apenas tabelas de conduta de **órgãos**
(entidades públicas). Racional completo em
[`docs/DECISAO_ESTADO_SOLICITANTE.md`](docs/DECISAO_ESTADO_SOLICITANTE.md).

Omitir o histórico é permitido: os campos valem `-1`, o `LightGBM` os trata como
faltantes e `/score` devolve `historico_informado: false` para que a operação
degradada não passe silenciosa.

---

# Tutorial: carregar o modelo no BentoML desde o zero

Sequência completa, do clone até a primeira resposta HTTP. Ambiente gerido por
**uv** (não use `pip` diretamente).

## Passo 1 — clonar e criar o ambiente

```bash
git clone git@github.com:lgallindo/lai-triagem.git && cd lai-triagem
uv sync --extra serve
```

`uv sync` lê o `pyproject.toml` e fixa as versões exatas do treinamento
registrado. O extra `serve` acrescenta `bentoml` e `pydantic`, necessários
apenas para servir.

## Passo 2 — conferir que o artefato está presente

O repositório **já traz o modelo treinado**, então não é preciso treinar para
servir:

```bash
ls -la artifacts/model_arrival.txt artifacts/preprocessor.json
```

O formato é texto nativo do LightGBM mais um acompanhante JSON — sem `pickle`,
portanto sem acoplamento de versão no carregamento.

## Passo 3 — registrar o modelo no repositório de modelos do BentoML

```bash
uv run python scripts/register_bento.py
```

Saída esperada:

```
registered: lai_triagem_arrival:<etiqueta gerada>
  path     : /tmp/bentoml-model-lai_triagem_arrival-...
  trees    : 98
  features : 30
```

A etiqueta (`tag`) muda a cada registro; o serviço usa `:latest`. Confirme:

```bash
uv run bentoml models list
```

## Passo 4 — subir o serviço

```bash
uv run bentoml serve service.py:LaiTriagem
```

O serviço sobe em `http://localhost:3000`. A documentação interativa fica em
`http://localhost:3000/docs`.

## Passo 5 — primeira chamada

Com o histórico do solicitante informado pelo chamador (precisão plena):

```bash
curl -sS -X POST http://localhost:3000/score -H 'Content-Type: application/json' -d '{"pedido":{"OrgaoDestinatario":"CC-PR – Casa Civil da Presidência da República","DataRegistro":"15/09/2026","Escolaridade":"Ensino Fundamental","UF_sol":"PE","n_pedidos_previos":80,"prev_reenc_solicitante":3,"prev_reenc_rate_solicitante":0.0375,"n_pedidos_previos_neste_orgao":12,"prev_reenc_neste_orgao":0,"n_orgaos_distintos_previos":14,"dias_desde_ultimo_pedido":5}}'
```

Resposta:

```json
{
  "probabilidade_reencaminhamento": 0.076869,
  "alerta": "BAIXO RISCO",
  "threshold": 0.155738,
  "orgao_conhecido": true,
  "orgao_rate_historica": 0.478643,
  "orgao_rate_movel_90d": 0.243032,
  "base_rate_coorte": 0.080268,
  "historico_informado": true
}
```

Sem o histórico, a mesma requisição devolve `0.353582` com
`"historico_informado": false` — quase cinco vezes mais alto. O órgão é o mesmo;
a diferença é toda o que se sabe sobre o solicitante.

## Passo 6 — comparar com a linha de base

```bash
curl -sS -X POST http://localhost:3000/score_baseline -H 'Content-Type: application/json' -d '{"pedido":{"OrgaoDestinatario":"CC-PR – Casa Civil da Presidência da República","DataRegistro":"15/09/2026"}}'
```

Retorna `0.478643`, a taxa histórica crua do órgão, contra `0.076869` do
modelo com histórico informado. A consulta não distingue estreante de veterano;
o modelo distingue, e é daí que vem a diferença de precisão@5% (24,79% contra
30,99% no teste maturado).

## Passo 7 (opcional) — sem BentoML

Para conferir o caminho de código sem subir servidor:

```bash
uv run python examples/minimal_predict.py
```

## Passo 8 (opcional) — retreinar

Só é necessário para reproduzir o treinamento. Baixe os dados (~38 MB) e rode:

```bash
mkdir -p data/raw data/interim
# 2022-2026 formam a coorte; 2012-2021 servem SÓ para datar o nascimento de cada
# órgão (sem eles, dias_desde_primeiro_pedido_do_orgao fica censurado em 2022).
for y in $(seq 2012 2026); do
  curl -sS -o data/raw/Pedidos_csv_$y.zip "https://dadosabertos-download.cgu.gov.br/FalaBR/Arquivos_FalaBR/Pedidos_csv_$y.zip"
  unzip -oq data/raw/Pedidos_csv_$y.zip -d data/interim/
done
uv run python scripts/train.py
```

Leva cerca de **40 s** no total (8 s de carga e featurização, ≈1,7 s por
ajuste, mais a datação dos órgãos). Procedimento completo e auditável em
[`docs/TREINAMENTO.md`](docs/TREINAMENTO.md).

---

# Entradas

Todos os campos abaixo existem no instante em que o pedido chega à caixa de
entrada do Fala.BR. `OrgaoDestinatario` e `DataRegistro` são obrigatórios; o
resto é opcional e ausência é tratada como valor faltante.

| Campo | Origem | Tipo | Uso no modelo | Razão da inclusão |
|---|---|---|---|---|
| `OrgaoDestinatario` | Pedidos | categórico | **11,56%** do ganho; alimenta as tabelas por órgão, que somam 60,4% | Órgão a que o cidadão endereçou. Sobreviveu à auditoria H2: é o endereçado, não o destinatário final |
| `DataRegistro` | Pedidos | data | deriva `reg_month`, `reg_dow`, `reg_day` | Único carimbo temporal disponível na chegada |
| `Esfera` | Pedidos | categórico | baixo | Federal/estadual/municipal; separa regimes de competência |
| `UF` | Pedidos | categórico | baixo | UF do pedido quando não federal |
| `Municipio` | Pedidos | categórico | 0,6% | Município do pedido quando não federal |
| `FormaResposta` | Pedidos | categórico | baixo | Escolhida pelo solicitante **na abertura** — logo, disponível |
| `OrigemSolicitacao` | Pedidos | categórico | 0,7% | Balcão SIC vs Internet; definido na abertura |
| `TipoDemandante` | Solicitantes | categórico | 0,6% | Pessoa física/jurídica |
| `Genero` | Solicitantes | categórico | baixo | Perfil; 70,7% ausente |
| `Escolaridade` | Solicitantes | categórico | baixo | Perfil; **76,2% ausente** — central na auditoria de equidade |
| `Profissao` | Solicitantes | categórico | 1,2% | Perfil; 76,9% ausente |
| `TipoPessoaJuridica` | Solicitantes | categórico | baixo | Vazio para pessoa física |
| `Pais` | Solicitantes | categórico | baixo | País de residência |
| `UF_sol` | Solicitantes | categórico | 1,0% | UF de residência; alimenta `uf_match` |
| `Municipio_sol` | Solicitantes | categórico | **11,5%** | Município de residência — terceiro sinal mais forte |
| `DataNascimento` | Solicitantes | data | deriva `idade` (2,5%) | Idade na data do registro; descartada fora de 10–110 anos |

## Variáveis derivadas — lado do órgão (embarcadas no artefato)

Conduta de entidade pública; sem dado pessoal. Reajustadas a cada retreinamento.

| Derivada | Fórmula | Ganho |
|---|---|---|
| `orgao_rate` | taxa histórica suavizada do órgão, ajustada **só nos anos de treino** (prior 50 × taxa-base) | **39,35%** |
| `orgao_rate_movel_90d` | taxa em janela móvel de 90 d, **estritamente anterior** à data do pedido | **16,13%** |
| `orgao_rate_movel_365d` | idem, 365 d | 3,62% |
| `dias_desde_primeiro_pedido_do_orgao` | idade do órgão; datada de 2012 em diante | 1,32% |
| `idade` | `DataRegistro − DataNascimento`, em anos | baixo |
| `reg_month`, `reg_dow`, `reg_day` | componentes de `DataRegistro` | baixo |
| `uf_match` | `UF_sol == UF` | baixo |

## Variáveis de histórico — informadas pelo chamador

Dado pessoal; **não embarcadas**. Todas opcionais, padrão `-1`.

| Campo | Significado | Ganho |
|---|---|---|
| `n_pedidos_previos_neste_orgao` | pedidos anteriores deste solicitante **a este órgão** | **6,66%** |
| `prev_reenc_rate_solicitante` | razão entre reencaminhados e total anteriores | 4,44% |
| `prev_reenc_neste_orgao` | reencaminhamentos anteriores deste solicitante neste órgão | 3,91% |
| `dias_desde_ultimo_pedido` | recência da última interação | 1,41% |
| `n_pedidos_previos` | total de pedidos anteriores (agregado) | 1,08% |
| `n_orgaos_distintos_previos` | amplitude: quantos órgãos distintos já acionou | 0,98% |
| `prev_reenc_solicitante` | contagem de reencaminhamentos anteriores | baixo |

Todas calculadas no treino com soma acumulada **deslocada**, de modo que a linha
corrente nunca vê a si mesma nem o futuro. Solicitante anonimizado
(`IdSolicitante == '0'`, 16,9% das linhas) não acumula histórico.

# Campos excluídos, e por quê

Toda exclusão é **empírica**, não precaucional. Os testes estão em
[`docs/VERIFICATION.md`](docs/VERIFICATION.md). O serviço **recusa** qualquer
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
| texto do pedido (`ResumoSolicitacao`, `DetalhamentoSolicitacao`) | na abertura | Disponível, mas **fora de escopo** pelo Termo de Abertura. Exige os arquivos `_Filtrado` (~80 MB/ano contra 7–9 MB) |

Também são removidas da modelagem as **459 linhas** com
`Situacao == "Encaminhada por Outro Órgão"`: estão em trânsito, de modo que seu
`OrgaoDestinatario` é o receptor, não o endereçado.

# Saídas

## `POST /score`

| Campo | Tipo | Significado |
|---|---|---|
| `probabilidade_reencaminhamento` | float 0–1 | Saída do LightGBM. **Não é calibrada** — serve para ordenar, não como probabilidade literal |
| `alerta` | `"ALTO RISCO"` \| `"BAIXO RISCO"` | Comparação com `threshold` |
| `threshold` | float | 0,1691 — ponto de operação da fila de 10% |
| `orgao_conhecido` | bool | `false` se o órgão não aparece nos anos de treino; nesse caso o escore recai na taxa-base |
| `orgao_rate_historica` | float | Taxa histórica do órgão, exposta para auditabilidade do escore |
| `base_rate_coorte` | float | 0,080268 — taxa-base da coorte de treino, para referência |

## `POST /score_baseline`

| Campo | Tipo | Significado |
|---|---|---|
| `probabilidade_reencaminhamento` | float | A taxa histórica do órgão, sem modelo |
| `alerta` | string | Mesmo limiar |
| `metodo` | string | `"lookup histórico por órgão (sem modelo)"` |
| `orgao_conhecido` | bool | Idem |

## `GET /health`

Retorna `model_tag`, `n_trees`, `n_features`, `features`,
`excluded_leakage_fields`, `train_years`, `data_snapshot`.

## Erro de vazamento

Enviar campo posterior à triagem devolve `ValueError`:

```
post-hoc field(s) supplied, refusing to score: ['FoiProrrogado'].
These are unavailable when a request arrives; see docs/VERIFICATION.md.
```

# Estrutura

| Caminho | Papel |
|---|---|
| [`docs/DECISAO_ESTADO_SOLICITANTE.md`](docs/DECISAO_ESTADO_SOLICITANTE.md) | Onde vive o histórico do solicitante e por quê — decisão de proteção de dados |
| [`docs/CAMPOS_POST_HOC.md`](docs/CAMPOS_POST_HOC.md) | O que é campo *post hoc*, por que não serve para treinar, e o protocolo de identificação |
| [`docs/TREINAMENTO.md`](docs/TREINAMENTO.md) | Procedimento de treinamento reproduzível e configuração do LightGBM |
| [`docs/VERIFICATION.md`](docs/VERIFICATION.md) | Auditoria de vazamento e viabilidade — **comece aqui** |
| `scripts/experiment_features_hpo.py` | Ablação de variáveis derivadas e busca rápida de hiperparâmetros |
| [`docs/LITERATURE.md`](docs/LITERATURE.md) | Varredura da literatura e protocolo |
| [`docs/VENUE.md`](docs/VENUE.md) | Plano de publicação, com links auditados |
| `scripts/train.py` | Treinamento com corte temporal: variantes, diagnóstico vazado e linha de base |
| `scripts/verify_leakage_v2.py` | H1 `AssuntoPedido`; H2 duplicatas e junção com Recursos |
| `scripts/verify_h2_final.py`, `verify_h2_baserate.py` | H2, teste direcional |
| `scripts/verify_h3_prazo.py` | H3 vazamento do prazo; H4 disponibilidade demográfica |
| `scripts/lit_scan*.py` | Varredura bibliográfica (OpenAlex/Crossref) |
| `lai_triagem/featurize.py` | Featurização de chegada e barreira de vazamento, compartilhada por treino e serviço |
| `service.py`, `scripts/register_bento.py` | Serviço BentoML e registro do modelo |
| `examples/minimal_predict.py` | Exemplo mínimo de escoragem |

*Os nomes de arquivos, classes, funções e variáveis permanecem em en_US; a
documentação e os comentários estão em pt_BR.*

# Limitações conhecidas

- **`Escolaridade` 76,2% ausente**, `Profissao` 76,9%. A regra de abstenção do
  Termo de Abertura — não pontuar quando falta perfil — recusaria **77,58%** dos
  pedidos, o que não é um produto viável.
- Variáveis demográficas somam ~5% do ganho; 76% é identidade do órgão.
- Rótulos de 2026 sofrem **censura à direita** (5,75% de positivos entre os
  maturados contra 3,60% nos recentes).
- As tabelas por órgão são um retrato do fim da janela de dados; exigem reajuste
  periódico, sem o qual `orgao_rate_movel_90d` envelhece e perde valor.
- 45,8% das linhas não têm histórico de solicitante aproveitável (16,9%
  anonimizadas, 28,9% de quem pediu uma vez só), então o ganho vem de pouco
  mais da metade do volume.
- O limiar 0,155738 é ponto de operação da fila de 10%, não probabilidade calibrada.

# Dados e licença

Fonte: CGU Dados Abertos,
<https://dadosabertos-download.cgu.gov.br/FalaBR/Arquivos_FalaBR/>, retrato
`20260914`. Usa os arquivos **sem texto** (~7–9 MB/ano).

Código sob **GPL-3.0-or-later** (veja [`LICENSE`](LICENSE)). Todas as
dependências são compatíveis: LightGBM e pydantic sob MIT; pandas, NumPy e
scikit-learn sob BSD-3; PyArrow e BentoML sob Apache-2.0.

Nenhum dado é versionado: **git puro**, sem DVC e sem git-lfs.
