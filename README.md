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
Todos os números de desempenho vivem em [`docs/METRICAS.md`](docs/METRICAS.md),
que é **gerado pelo treinamento** — este README não os repete, para não
envelhecer. O quadro no fecho desta seção:

| Escore, teste 2026 maturado | PR-AUC | prec@5% |
|---|---|---|
| Consulta histórica por órgão (sem modelo) | 0,1641 | **24,80%** |
| LightGBM | 0,1836 | 24,67% |

**O modelo NÃO supera a consulta por órgão na métrica primária.** Perde 0,5% em
precisão@5% — a fila que o produto de fato entrega. Ganha 11,9% em PR-AUC e
10,9% em precisão@10%, o que não compensa.

A explicação está no ganho por variável: `orgao_rate` 40,51% +
`OrgaoDestinatario` 20,19% + as duas taxas móveis 13,01% somam **73,71% de
identidade do órgão**. Todo o histórico do solicitante soma **7,98%**. Contando
também a idade do órgão, a identidade do órgão chega a 76,34%. O modelo é,
essencialmente, a tabela de consulta com enfeites.

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
> [`docs/auditorias/`](docs/auditorias/INDICE.md).

**A conclusão original do projeto volta a valer:** com variáveis honestas de
chegada, praticamente todo o sinal recuperável é "alguns órgãos são
cronicamente mal endereçados", e uma tabela de consulta de uma linha captura
isso.

Efeito colateral notável da defasagem de maturação:
`orgao_rate_movel_90d` caiu para **2,77%** do ganho, enquanto a janela de 365
dias ficou em **7,50%**. Defasada em 60 dias, uma janela de 90 dias fica quase
toda obsoleta — perde exatamente a atualidade que a justificava.

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
contaminação detectável. Remover as variáveis também não mudaria nada
(0,1836 contra 0,1853). Detalhes em `scripts/experiment_h6_mitigacao.py`.

O dado pessoal que o serviço recebe — perfil e contadores de histórico — **não é
retido no artefato**. Ver
[`docs/DECISAO_ESTADO_SOLICITANTE.md`](docs/DECISAO_ESTADO_SOLICITANTE.md).

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

> **Se você já recebeu a pasta pronta, pule o `git clone`** e vá direto ao
> `uv sync`. Clonar dentro de uma cópia existente cria
> `lai-triagem/lai-triagem/` sem dar erro, e confunde.

```bash
git clone git@github.com:lgallindo/lai-triagem.git && cd lai-triagem
uv sync --extra serve
```

`uv` é o gerenciador de pacotes e de ambiente virtual usado aqui. `uv sync` lê o
`pyproject.toml` e instala as versões exatas do treinamento registrado; o extra
`serve` acrescenta `bentoml` e `pydantic`, necessários apenas para servir.

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
  trees    : <varia>
  features : <varia>
```

> **As contagens de `trees` e `features` mudam a cada retreinamento** — não
> compare com um número fixo. Os valores correntes estão em
> [`docs/METRICAS.md`](docs/METRICAS.md), que é gerado pelo treinamento.

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

**Se aparecer `OSError: [Errno 98] Address already in use`:** a porta 3000 já
está ocupada, quase sempre por um serviço que você mesmo deixou rodando antes.
Descubra quem é e encerre:

```bash
ss -ltnp | grep :3000          # mostra o processo que está na porta
pkill -f "bentoml serve"       # encerra um serviço anterior
```

Ou use outra porta:

```bash
uv run bentoml serve service.py:LaiTriagem --port 3001
```

**Para confirmar que subiu**, antes de tentar o Passo 5:

```bash
curl -sS http://localhost:3000/healthz && echo " -> de pé"
```

## Passo 5 — primeira chamada

Com o histórico do solicitante informado pelo chamador (precisão plena). São
**oito** campos de histórico, e é tudo-ou-nada: mandar sete faz o serviço
descartar o conjunto inteiro. Note os dois terminados em `_den`, que são os
denominadores maturados — eles **não** podem faltar:

```bash
curl -sS -X POST http://localhost:3000/score -H 'Content-Type: application/json' -d '{"pedido":{"OrgaoDestinatario":"CC-PR – Casa Civil da Presidência da República","DataRegistro":"15/09/2026","Escolaridade":"Ensino Fundamental","UF_sol":"PE","n_pedidos_previos":80,"prev_reenc_solicitante":3,"prev_reenc_solicitante_den":80,"n_pedidos_previos_neste_orgao":12,"prev_reenc_neste_orgao":0,"prev_reenc_neste_orgao_den":12,"n_orgaos_distintos_previos":14,"dias_desde_ultimo_pedido":5}}'
```

Resposta:

```json
{
  "probabilidade_reencaminhamento": 0.408009,
  "alerta": "ALTO RISCO",
  "threshold": 0.162928,
  "probabilidade_calibrada": 0.312586,
  "calibrada_apenas_para_leitura": true,
  "orgao_conhecido": true,
  "orgao_rate_historica": 0.478643,
  "orgao_rate_movel_90d": 0.257018,
  "base_rate_coorte": 0.080268,
  "historico_informado": true,
  "historico_parcial_ignorado": false
}
```

Confira sempre `historico_informado`. Se vier `false` com
`historico_parcial_ignorado: true`, faltou algum dos oito campos e o escore que
você recebeu é o de **sem histórico**, não o de precisão plena.

E **sem** o histórico — só os dois campos obrigatórios:

```bash
curl -sS -X POST http://localhost:3000/score -H 'Content-Type: application/json' -d '{"pedido":{"OrgaoDestinatario":"CC-PR – Casa Civil da Presidência da República","DataRegistro":"15/09/2026"}}'
```

O escore **cai** de 0,408009 para 0,337359 e `historico_informado` vira
`false`. O órgão é o mesmo; a diferença é tudo o que se sabe sobre o
solicitante — e, neste caso, o histórico deste veterano **agravava** o risco.

Curiosidade que justifica a calibração ser só para leitura: nos três cenários
`probabilidade_calibrada` devolve o mesmo 0,312586, porque a regressão isotônica
tem um platô cobrindo essa faixa. Ela empata casos que o escore cru separa — por
isso a fila é ordenada pelo escore cru.

> **O histórico é um conjunto de tudo-ou-nada.** Se você mandar alguns dos oito
> campos e não todos, o serviço **descarta o conjunto inteiro** e avisa em
> `historico_parcial_ignorado: true`. Meio histórico produziria uma combinação
> que nunca aparece no treinamento.

## Passo 6 — comparar com a linha de base

```bash
curl -sS -X POST http://localhost:3000/score_baseline -H 'Content-Type: application/json' -d '{"pedido":{"OrgaoDestinatario":"CC-PR – Casa Civil da Presidência da República","DataRegistro":"15/09/2026"}}'
```

Retorna `0.478643`, a taxa histórica crua do órgão. O modelo distingue
estreante de veterano e a consulta não — mas, após a correção do vazamento, essa
distinção **não se traduz** em ganho de precisão@5% (24,80% da base contra
24,67% do modelo). Mantemos o endpoint porque a comparação é o achado.

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
| `DataRegistro` | Pedidos | data, **`dd/mm/aaaa`** | deriva `reg_month`, `reg_dow`, `reg_day` | Único carimbo temporal disponível na chegada. Formato obrigatório: `15/09/2026` |
| `Esfera` | Pedidos | categórico | baixo | Federal/estadual/municipal; separa regimes de competência |
| `UF` | Pedidos | categórico | baixo | UF do pedido quando não federal |
| `Municipio` | Pedidos | categórico | 0,36% | Município do pedido quando não federal |
| `FormaResposta` | Pedidos | categórico | baixo | Escolhida pelo solicitante **na abertura** — logo, disponível |
| `OrigemSolicitacao` | Pedidos | categórico | 0,53% | Balcão SIC vs Internet; definido na abertura |
| `TipoDemandante` | Solicitantes | categórico | 0,31% | Pessoa física/jurídica |
| `Genero` | Solicitantes | categórico | baixo | Perfil; 70,7% ausente |
| `Escolaridade` | Solicitantes | categórico | baixo | Perfil; **76,2% ausente** — central na auditoria de equidade |
| `Profissao` | Solicitantes | categórico | 0,62% | Perfil; 76,9% ausente |
| `TipoPessoaJuridica` | Solicitantes | categórico | baixo | Vazio para pessoa física |
| `Pais` | Solicitantes | categórico | baixo | País de residência |
| `UF_sol` | Solicitantes | categórico | 0,56% | UF de residência; alimenta `uf_match` |
| `Municipio_sol` | Solicitantes | categórico | **8,73%** | Município de residência — quarto sinal mais forte |
| `DataNascimento` | Solicitantes | data | deriva `idade` (1,13%) | Idade na data do registro; descartada fora de 10–110 anos |

## Variáveis derivadas — lado do órgão (embarcadas no artefato)

Conduta de entidade pública; sem dado pessoal. Reajustadas a cada retreinamento.

| Derivada | Fórmula | Ganho |
|---|---|---|
| `orgao_rate` | taxa histórica suavizada do órgão, ajustada **só nos anos de treino** (prior 50 × taxa-base) | **40,51%** |
| `orgao_rate_movel_365d` | taxa em janela móvel de 365 d, **defasada 60 d** | **10,20%** |
| `orgao_rate_movel_90d` | idem, 90 d — a defasagem de maturação a esvaziou | 2,81% |
| `dias_desde_primeiro_pedido_do_orgao` | idade do órgão; datada de 2012 em diante | 2,63% |
| `idade` | `DataRegistro − DataNascimento`, em anos | baixo |
| `reg_month`, `reg_dow`, `reg_day` | componentes de `DataRegistro` | baixo |
| `uf_match` | `UF_sol == UF` | baixo |

## Variáveis de histórico — informadas pelo chamador

Dado pessoal; **não embarcadas** no artefato. São **oito campos, e o conjunto
é atômico**: ou você manda os oito, ou o serviço descarta todos e usa o padrão
`-1`. Não existe "mandar alguns". Meio histórico produziria uma combinação de
valores que nunca aparece no treinamento, e o escore não teria significado.

| Campo | Significado | Ganho |
|---|---|---|
| `n_pedidos_previos_neste_orgao` | pedidos anteriores deste solicitante **a este órgão** | **3,34%** |
| `n_pedidos_previos` | total de pedidos anteriores (agregado) | 1,92% |
| `n_orgaos_distintos_previos` | amplitude: quantos órgãos distintos já acionou | 1,28% |
| `dias_desde_ultimo_pedido` | recência da última interação; `-1` se não houver | 1,19% |
| `prev_reenc_solicitante` | reencaminhamentos anteriores, **numerador** | 0,10% |
| `prev_reenc_neste_orgao` | idem, restrito a este órgão, **numerador** | 0,01% |
| `prev_reenc_solicitante_den` | **denominador maturado** do numerador acima | — |
| `prev_reenc_neste_orgao_den` | **denominador maturado**, restrito a este órgão | — |

Os dois denominadores não recebem ganho próprio porque não entram no modelo
como variáveis: o serviço usa cada par numerador/denominador para calcular
internamente as razões `prev_reenc_rate_solicitante` (0,11% de ganho) e
`prev_reenc_rate_neste_orgao` (0,03%). Elas **não** devem ser enviadas pelo
chamador — eram entradas em versões anteriores e deixaram de ser quando a
maturação de 60 dias passou a ser aplicada ao denominador.

Por que denominador separado, e não a razão pronta: o denominador só conta
pedidos cujo desfecho já **maturou** (60 dias). Enviar a razão pronta deixaria o
chamador escolher, sem saber, um denominador não maturado — e reintroduziria
pelo contrato o vazamento que a correção 2 eliminou no treinamento.

Solicitante anonimizado (`IdSolicitante == '0'`, 16,9% das linhas) não acumula
histórico.

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
| `ProtocoloPedido` / `protocolo_seq` | atribuído na abertura, mas pela **unidade registradora** | A fatia `[5:11]` do protocolo não é um sequencial neutro: separa **79×** dentro de um mesmo órgão-ano (INSS 2022: 28,38% no 1º quarto contra 0,36% no 4º). Codifica qual unidade registrou, cujo comportamento de encaminhamento é quase determinístico (H5) |
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
| `threshold` | float | o valor em [`docs/METRICAS.md`](docs/METRICAS.md) — quantil 90 dos escores de validação, ponto de operação da fila de 10%. Reajustado a cada retreinamento |
| `orgao_rate_movel_90d` | float | Taxa do órgão na janela móvel de 90 d, exposta para auditoria |
| `historico_informado` | bool | `false` se o chamador omitiu o histórico do solicitante — o escore está degradado |
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

## `POST /health`

**É POST, não GET** — todo endpoint do BentoML declarado com `@bentoml.api` é
POST, e um `GET /health` devolve HTTP 405. Para a sonda de liveness use
`GET /healthz`, que é do próprio BentoML:

```bash
curl -sS -X POST http://localhost:3000/health -H 'Content-Type: application/json' -d '{}'
```

Retorna `model_tag`, `n_trees`, `n_features`, `features`, `threshold`,
`queue_fraction`, `caller_supplied_features`, `caller_supplied_default`,
`caller_supplied_rationale`, `embedded_organ_tables`, `contains_personal_data`,
`excluded_leakage_fields`, `train_years` e `data_snapshot`. É a forma de
descobrir o contrato dos oito campos sem ler o código.

## Erro de vazamento

Enviar campo posterior à triagem devolve `ValueError`:

```
post-hoc field(s) supplied, refusing to score: ['FoiProrrogado'].
These are unavailable when a request arrives; see docs/VERIFICATION.md.
```

# Estrutura

| Caminho | Papel |
|---|---|
| [`docs/METRICAS.md`](docs/METRICAS.md) | **Gerado** por train.py; fonte única de todo número de desempenho |
| [`docs/DECISAO_ESTADO_SOLICITANTE.md`](docs/DECISAO_ESTADO_SOLICITANTE.md) | Onde vive o histórico do solicitante e por quê — decisão de proteção de dados |
| [`docs/auditorias/`](docs/auditorias/INDICE.md) | Auditorias, um arquivo por data: o que outros agentes — e o roteiro mecânico — encontraram neste trabalho |
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
- Variáveis demográficas somam pouco; **73,71% do ganho é identidade do
  órgão**. E elas vêm de um retrato atual do cadastro, não do perfil na
  abertura do pedido — limitação H6, sem efeito mensurável medido.
- **O ganho por variável é mantido à mão neste README e já divergiu três
  vezes.** Os valores acima foram medidos em 18/09/2026 contra
  `artifacts/model_arrival.txt`. Gerá-los junto de
  [`docs/METRICAS.md`](docs/METRICAS.md) é a correção estrutural pendente.
- Rótulos de 2026 sofrem **censura à direita** (5,75% de positivos entre os
  maturados contra 3,60% nos recentes).
- As tabelas por órgão são um retrato do fim da janela de dados; exigem reajuste
  periódico, sem o qual `orgao_rate_movel_90d` envelhece e perde valor.
- 45,8% das linhas não têm histórico de solicitante aproveitável (16,9%
  anonimizadas, 28,9% de quem pediu uma vez só), então o ganho vem de pouco
  mais da metade do volume.
- O limiar é ponto de operação da fila de 10%, não probabilidade calibrada, e
  muda a cada retreinamento. Valor corrente em
  [`docs/METRICAS.md`](docs/METRICAS.md).

# Dados e licença

Fonte: CGU Dados Abertos,
<https://dadosabertos-download.cgu.gov.br/FalaBR/Arquivos_FalaBR/>, retrato
`20260914`. Usa os arquivos **sem texto** (~7–9 MB/ano).

Código sob **GPL-3.0-or-later** (veja [`LICENSE`](LICENSE)). Todas as
dependências são compatíveis: LightGBM e pydantic sob MIT; pandas, NumPy e
scikit-learn sob BSD-3; PyArrow e BentoML sob Apache-2.0.

Nenhum dado é versionado: **git puro**, sem DVC e sem git-lfs.

---

# Glossário

Termos que este README usa e que não são óbvios no primeiro curso de aprendizado
de máquina. Em ordem de aparição, não alfabética.

| Termo | O que é, em uma frase |
|---|---|
| **BentoML** | Ferramenta que embala um modelo treinado e o expõe como serviço HTTP |
| **`uv`** | Gerenciador de pacotes e ambientes virtuais Python; substitui `pip` + `venv` |
| **artefato** | Os arquivos que saem do treinamento e são usados para pontuar: aqui, o modelo em texto mais o JSON acompanhante |
| **taxa-base** | Proporção de positivos no conjunto. Se 5,75% dos pedidos são reencaminhados, a taxa-base é 5,75% — é o que você acertaria chutando ao acaso |
| **coorte** | O conjunto de pedidos usado no estudo (aqui, 2022 a 2026) |
| **precisão@5%** | Dos 5% de pedidos que o modelo considera mais arriscados, que fração realmente foi reencaminhada. É **a** métrica que importa aqui, porque o produto é uma fila com capacidade limitada |
| **PR-AUC** | Resume a qualidade do modelo em todos os tamanhos de fila de uma vez, num número entre 0 e 1. Mais alto é melhor. Com poucos positivos, é mais informativa que a acurácia |
| **ROC-AUC** | Outra medida resumo, entre 0 e 1; 0,5 é o acaso. Tende a parecer boa mesmo quando o modelo é ruim em dados desbalanceados, então aqui é secundária |
| **ganho (de variável)** | Quanto uma variável contribuiu para as decisões da árvore, em % do total. Não é causalidade, é peso no ajuste |
| **ganho / lift (de fila)** | Quantas vezes o modelo é melhor que o acaso naquela fila. "4,3×" significa quatro vezes mais acertos que escolher aleatoriamente |
| **pp** | Pontos percentuais. Sair de 20% para 22% é +2 pp (e não +2%) |
| **limiar** (*threshold*) | O valor de corte acima do qual o pedido é marcado ALTO RISCO |
| **quantil 90** | O valor que 90% dos escores não ultrapassam. Usamos como limiar para que a fila fique com os 10% mais arriscados |
| **calibrado** | Um escore calibrado pode ser lido como probabilidade de verdade ("0,30" ≈ 30% de chance). O escore cru **não** pode: ele só serve para ordenar |
| **codificação de alvo** (*target encoding*) | Trocar uma categoria pela taxa histórica do que se quer prever naquela categoria. É o que `orgao_rate` faz: cada órgão vira sua própria taxa de reencaminhamento |
| **suavização com prior** | Ao calcular a taxa de um órgão com poucos pedidos, misturar com a taxa geral para não confiar em amostra pequena. "Prior 50" = equivale a acrescentar 50 pedidos médios |
| **vazamento** (*data leakage*) | Usar, para prever, uma informação que na hora real da decisão ainda não existiria. Faz o modelo parecer ótimo no teste e falhar em produção |
| ***post hoc*** | Campo preenchido **depois** do momento da decisão. Fonte mais comum de vazamento |
| **censura à direita** | Um pedido recente pode ainda vir a ser reencaminhado; o "não" dele é provisório. Por isso só avaliamos pedidos com pelo menos 60 dias |
| **maturação** | O tempo que se espera antes de confiar no rótulo de um pedido. Aqui, 60 dias |
| **H1 … H6** | As seis hipóteses testadas sobre os dados, cada uma sobre uma variável poder ou não ser usada. O resumo está em [`docs/VERIFICATION.md`](docs/VERIFICATION.md) |
| **featurização** | Transformar os campos brutos nas variáveis que o modelo consome |
| **ablação** | Remover uma variável de propósito para medir o quanto ela valia |
| **HPO** | Busca de hiperparâmetros: testar configurações do algoritmo para ver qual vai melhor |
| **linha de base** (*baseline*) | O jeito mais simples de resolver o problema, contra o qual o modelo tem de provar valor. Aqui: ordenar os pedidos pela taxa histórica do órgão, sem modelo nenhum |
