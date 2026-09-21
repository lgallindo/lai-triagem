# Adequação a boas práticas — 21/09/2026

Não é auditoria: é a faxina que precede a comparação entre Temporian e
Featuretools. Se o núcleo continuasse duplicado, cada branch bifurcaria a
própria cópia e a comparação mediria os defeitos daqui, não as bibliotecas.

Fez-se em `boas-praticas`, integrada por `develop`.

## O defeito vivo, achado no levantamento

`_clean` estava definida em **seis** arquivos. A correção do H9, de 18/09,
alcançou **uma**: `scripts/train.py`. Quatro ficaram com a versão que não
limpava nada, e uma sexta tinha uma terceira implementação.

O grave não é a dívida técnica. `scripts/refresh_organ_tables.py` **não é
experimento**: é a ferramenta de produção que reescreve as tabelas de órgão
dentro do artefato, sem retreinar. Executá-la devolveria as chaves com espaço
ao `preprocessor.json` e quebraria de novo, em silêncio, os 41 órgãos que o
serviço passara a encontrar.

**A correção do H9 tinha prazo de validade de três dias, e ninguém saberia.**

Nota sobre método: a primeira checagem acusou `train.py` como defeituoso e
`verify_leakage_v2.py` como correto — o oposto da verdade. Ela casava com o
*docstring* onde eu documentara o defeito antigo, não com o código.

## O que mudou

| | antes | depois |
|---|---|---|
| Achados do `ruff` | **227** | **0** |
| Erros do `mypy` | não rodava | **0** |
| Cópias de `_clean` | 6, em 3 versões | **1** |
| Definições de `PRIOR_MOVEL` | 2, "mantenha iguais" | **1** |
| `Path.home() / "lai-triagem"` | **11 arquivos** | **0** |
| Configuração de lint/tipo/teste | nenhuma | `pyproject.toml` |

Novo pacote: `lai_triagem/{config,dados,metricas,codificacao}.py`.

### `ruff`

Limite de 100 colunas, e não as 88 padrão: medido, a 88 seriam 183 violações e
a 100 são 30. Reformatar 183 linhas que não estão erradas esconderia as
correções de verdade.

O achado que importava foi **`B905`, `zip()` sem `strict=`** — que trunca em
silêncio quando os comprimentos divergem, a mesma família do H7. O autofix do
ruff põe `strict=False`, que cala o aviso sem remover o risco; os seis foram
julgados um a um. Em `check_artefato.py` ficou `strict=False` **de propósito**,
com comentário: ali a diferença de comprimento é justamente o que se relata.

`C408` ficou desligada com justificativa escrita: os usos são listas de
parâmetro, e `dict(sep=";", ...)` lê-se melhor que o literal com aspas. As
outras 41 foram corrigidas, não silenciadas.

### `mypy`, e o defeito real que ele achou

`ler()` passava o `Path | None` de `arquivo_mais_recente` direto para
`read_csv`. Se o arquivo do ano não existisse, o erro sairia de dentro do
pandas, obscuro — e é **exatamente** o modo de falha que já mordeu este
projeto: quando a CGU trocou o prefixo do retrato, arquivos sumiram em silêncio
e a idade dos órgãos ficou censurada em 2022 sem nada reclamar. Agora levanta
`FileNotFoundError` dizendo qual arquivo faltou e onde procurou.

Um único `type: ignore`, justificado no lugar: `read_csv` é sobrecarregada e o
mypy não casa `**kwargs` desempacotado contra sobrecargas.

## H10 — o maior ganho de todo o processo

`orgao_rate` é codificação de alvo, e **não usava cross-fitting**: a taxa saía
do treino inteiro e voltava para o próprio treino, de modo que cada linha
carregava o próprio rótulo na maior variável do modelo.

Não era vazamento de teste. O dano era o modelo **confiar demais** numa
variável que, no treino, era melhor do que jamais seria em produção.

Medido com a fórmula de suavização idêntica nos dois braços:

| | PR-AUC | precisão@5% | contra a base |
|---|---|---|---|
| sem cross-fitting | 0,1789 | 21,79% | −3,01 pp |
| **com cross-fitting** | **0,1901** | **24,23%** | **−0,57 pp** |

A diferença na variável é minúscula — média 0,0022, e só 9.923 de 391.425
linhas mudam mais de 0,01. Mudança pequena na entrada produzindo mudança grande
no resultado é a assinatura de um modelo apoiado em variável otimista.

O sinal mais claro: **o modelo caiu de 160 para 56 árvores**. Não precisava
mais daquela profundidade.

**A conclusão não muda** — na métrica que o produto entrega, a tabela de
consulta continua à frente. Mas a distância encolheu de −12,1% para **−2,3%**,
e isso merece ser dito com todas as letras.

## E as guardas pegaram uma divergência entre duas ferramentas minhas

O atualizador automático grava `0.31659` (o `json.dumps` corta o zero à
direita) e o verificador procurava `0.316590`. Mesmo número, textos
diferentes. A regra de localidade passa a comparar **numericamente**.

## Prova de que a refatoração não mudou comportamento

Antes do H10, o treino completo foi reexecutado e
`artifacts/model_arrival.txt` saiu **idêntico bit a bit** ao commitado. A
deduplicação, os caminhos e o ruff não moveram um único escore. O que mudou o
modelo foi, e só, o H10 — deliberado e medido.
