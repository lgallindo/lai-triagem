# Varredura da literatura — aprendizado de máquina no tratamento de pedidos de acesso à informação

**Execução:** 15–16/09/2026. **Data de auditoria de todos os links:** 16/09/2026.

## 1. Protocolo (o que foi efetivamente executado)

Reproduzível por `scripts/lit_scan.py`, `scripts/lit_scan_v2.py` e
`scripts/lit_scan_lai.py`.

| Etapa | Ferramenta | Detalhe |
|---|---|---|
| S1 | OpenAlex `search=` | 10 consultas amplas, 8 resultados cada, deduplicadas por título e ordenadas por citações |
| S2 | OpenAlex `filter=title_and_abstract.search:` | 5 grupos temáticos, 15 sondagens precisas — **contagem zero aqui é evidência**, não ruído |
| S3 | OpenAlex, específico da LAI | 10 sondagens: `Fala.BR`, `Lei de Acesso a Informacao`, `FOIA request prediction`, `government information request misrouting`, … |
| S4 | Crossref REST | Verificação em nível de DOI de toda obra citada abaixo |
| S5 | Auditoria HTTP | `audit_links.py` / `audit_links2.py` — status e cadeia de redirecionamento por URL |

Duas ressalvas metodológicas, declaradas porque limitam a conclusão:

- **O Semantic Scholar ficou indisponível** (HTTP 429 do início ao fim); apenas
  OpenAlex e Crossref foram usados. Uma passagem por Scopus/WoS reforçaria a
  afirmação negativa.
- **A ordenação por relevância do OpenAlex é ruim neste tema.** A consulta
  `"freedom of information request machine learning"` restrita a título e resumo
  devolve **68 obras**, das quais *nenhuma* trata de processamento de pedidos de
  acesso — os primeiros resultados são reconhecimento facial térmico e
  comentários sobre o RGPD. Contagens isoladas são portanto inconfiáveis, e cada
  conjunto de resultados foi lido.

## 2. Achado principal: a literatura direta é inexistente

Nenhum trabalho publicado foi encontrado que preveja **risco de roteamento
incorreto ou de retrabalho** para pedidos de acesso à informação, no Brasil ou
fora dele.

Sondagens com resultado zero (S2/S3), cada uma um indicador de lacuna:

- `ouvidoria classificacao automatica` → **0**
- `misrouting administrative requests prediction` → **0**
- `Fala.BR` → nada no tema
- `reencaminhamento pedidos acesso informacao` → nada no tema

## 3. Agrupamentos adjacentes (o que existe)

### 3a. Classificação de demandas e reclamações de cidadãos — roteamento por tema, não risco de retrabalho

Os vizinhos mais próximos classificam *sobre o que* é um pedido, para então
roteá-lo. Nenhum modela a **probabilidade de o roteamento falhar**, que é o alvo
deste artefato.

| Obra | Ano | Veículo | Nota |
|---|---|---|---|
| [Structure of 311 service requests as a signature of urban location](https://doi.org/10.1371/journal.pone.0186314) | 2017 | PLOS ONE | Verificada no Crossref. Demandas 311 como sinal urbano; descritivo, não preditivo de roteamento incorreto |
| Intelligent Ombudsman: An AI-Based Approach to Demand Classification | 2026 | Springer LNCS | Análogo institucional mais próximo (classificação de demandas de ouvidoria); 0 citações, muito recente |
| Fine-Tuning of a LLM for Public Complaint Classification and Routing | 2026 | Iconic Research and Engineering Journals | Roteamento com LLM; veículo de baixa visibilidade |
| CivicFix: Smart Complaint Routing for Urban Solutions | 2025 | IJARCCE | Roteamento de reclamações municipais |
| Automated Classification of Arabic client Inquiries for Government Services | 2025 | — | Classificação de consultas a serviços públicos, por ajuste fino |

### 3b. Apoio algorítmico à decisão no setor público — a literatura de enquadramento

Diretamente relevante ao desenho do Termo de Abertura, em que um analista *vê um
sinalizador* e decide:

| Obra | Ano | Veículo | Citações | Por que importa aqui |
|---|---|---|---|---|
| [Human–AI Interactions in Public Sector Decision Making: "Automation Bias" and "Selective Adherence"](https://doi.org/10.1093/jopart/muac007) | 2022 | JPART | 423 | O risco central de um produto do tipo "sinalizador de prioridade": o analista adere em excesso ao escore |
| [Accountable Artificial Intelligence: Holding Algorithms to Account](https://doi.org/10.1111/puar.13293) | 2020 | Public Administration Review | 601 | Marco de responsabilização para um escore operado pelo Estado |
| [Fairness and Accountability Design Needs for Algorithmic Support in High-Stakes Public Sector Decision-Making](https://doi.org/10.1145/3173574.3174014) | 2018 | ACM CHI | 464 | Requisitos de desenho exatamente para esta classe de ferramenta |

As três foram verificadas no Crossref. As páginas das editoras devolvem HTTP 403
a clientes automatizados; os DOIs resolvem.

**Correção em relação a um rascunho anterior desta varredura:** um artigo sobre
perfilamento em múltiplos estágios e o art. 22 do RGPD foi citado contra o DOI
`10.1093/idpl/ipab002`. O Crossref mostra que esse DOI pertence a *"Data-driven
measures to mitigate the impact of COVID-19 in South America"* (International
Data Privacy Law, 2021). A citação estava errada e foi removida, em vez de
adivinhada.

### 3c. Análogo metodológico — triagem limitada por capacidade

Aprendizado de máquina em triagem de emergência hospitalar é o precedente
*metodológico* mais próximo de uma fila por precisão@k sob capacidade fixa de
revisores; por exemplo Raita et al. 2019 (*Critical Care*) e Hong et al. 2018
(*PLoS ONE*). Citados pelo método, não pelo domínio.

## 4. A lacuna que este artefato ocupa

Três contribuições, em ordem decrescente de originalidade:

1. **Originalidade do alvo.** Prever *retrabalho administrativo*
   (`FoiReencaminhado`) em vez do rótulo de roteamento. Nenhum trabalho anterior
   localizado.
2. **Um protocolo de auditoria de vazamento para dados abertos
   administrativos.** Três das quatro famílias candidatas de variáveis se
   revelaram posteriores ou inutilizáveis, e o teste decisivo foi diferente em
   cada caso: decaimento por recência para `AssuntoPedido`; argumento direcional
   de taxa por órgão para `OrgaoDestinatario`; assinatura estatutária de +10 dias
   para `prazo_dias`. Artigos publicados de aprendizado de máquina sobre dados
   administrativos raramente relatam isso, e o caso do `prazo_dias` mostra por
   quê: infla a PR-AUC em **+22 pp** e parece uma excelente variável.
3. **Um resultado negativo digno de publicação.** Removido o vazamento, uma
   consulta por órgão de uma única linha empata com o LightGBM (precisão@5%
   24,79% contra 24,44%). A auditoria de equidade também quantifica a
   "discriminação positiva" prevista pelo Termo de Abertura: solicitantes com
   `Ensino Fundamental` aparecem na fila de alerta dos 10% a **2,40×** sua
   participação populacional.

As contribuições 2 e 3 são independentes do veículo e não dependem de o modelo
funcionar — o que, dado o achado, é justamente o ponto.
