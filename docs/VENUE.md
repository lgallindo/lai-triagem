# Proposta de veículo de publicação — submeter até janeiro de 2027

**Data de auditoria dos links: 16/09/2026.** Toda URL abaixo foi efetivamente
requisitada (`audit_links.py`, `audit_links2.py`); o status é reproduzido tal
como retornou. `403` significa que a editora bloqueia clientes automatizados — a
página abre em navegador, e o DOI foi verificado à parte no Crossref.
**Prazos não foram auditados** e precisam ser confirmados na chamada oficial de
cada veículo; as chamadas de 2027 estavam em grande parte não publicadas na data
da auditoria.

---

## Sumário executivo

O ativo publicável **não é o modelo** — é a **auditoria de vazamento** e o
**resultado negativo**. Três das quatro famílias candidatas de variáveis se
mostraram posteriores à triagem ou inutilizáveis e, removidas elas, um
`groupby` de uma linha empata com um LightGBM ajustado (precisão@5% 24,79%
contra 24,44% no teste maturado de 2026). Isso configura um artigo de método e
advertência sobre aprendizado de máquina em dados abertos administrativos, não
um artigo de desempenho.

Isso determina a escolha do veículo. Congressos de aprendizado de máquina
orientados a desempenho não têm espaço para "nosso modelo não superou uma tabela
de consulta". Veículos que premiam rigor metodológico, resultados negativos e
realismo de implantação no setor público têm.

**Recomendação: duas submissões a partir de um único artefato.**

1. **ACM FAccT 2027** — principal. Apoio algorítmico à decisão em governo, com
   auditoria de equidade (`Ensino Fundamental` sinalizado a 2,40× da
   participação populacional) e uma recusa documentada de implantar modelo
   vazado. O FAccT valoriza explicitamente resultados negativos e críticos.
   Historicamente tem prazo de resumo no **fim de janeiro**, o que casa com a
   restrição, mas é também o principal risco.
2. **Revista do Serviço Público (ENAP)** — paralela, em português, **fluxo
   contínuo**, sem risco de prazo. Alcança o público que pode agir sobre o
   resultado: CGU e SICs federais. Uma revista de fluxo contínuo neutraliza a
   aposta de calendário do FAccT.

**Alternativa se o prazo do FAccT escorregar:** `dg.o 2027` (ACM Digital
Government Research), mesma comunidade, chamada tipicamente um pouco mais tarde.

**Não almejar:** BRACIS, SBBD, KDMiLe, ECML PKDD. Todos premiam desempenho
preditivo, que este artefato deliberadamente não entrega.

---

## Lista ordenada

### Faixa 1 — recomendados

| Veículo | URL | Auditoria | Adequação |
|---|---|---|---|
| **ACM FAccT 2027** | <https://facctconference.org/> | **200 OK** | Escoragem algorítmica no setor público, auditoria de equidade, resultados negativos no escopo. Prazo historicamente no fim de janeiro — **confirmar antes** |
| **Revista do Serviço Público (ENAP)** | <https://revista.enap.gov.br/index.php/RSP> | **200 OK** | pt-BR, fluxo contínuo, público praticante é a própria CGU/SIC. Sem risco de prazo |

### Faixa 2 — alternativas fortes

| Veículo | URL | Auditoria | Adequação |
|---|---|---|---|
| **dg.o 2027** (ACM Digital Government) | <https://dgsociety.org/> · [dg.o 2026](https://dgsociety.org/dgo-2026/) | **200 OK** (ambos) | Melhor encaixe puro em governo digital; anais ACM. Usar como alternativa ao FAccT |
| **Data & Policy** (Cambridge) | <https://www.cambridge.org/core/journals/data-and-policy> | 429 (limitação de taxa; ativo) | Acesso aberto, voltado a políticas públicas, fluxo contínuo. Publica explicitamente achados negativos e de implementação |
| **Revista de Administração Pública** (FGV) | <https://periodicos.fgv.br/rap> | **200 OK** | Principal revista brasileira de administração pública, fluxo contínuo |
| **AIES 2027** (AAAI/ACM AI Ethics & Society) | <https://www.aies-conference.com/> → `/2026/` | **200 OK** | O enquadramento ético serve; prazo em geral por volta de março, **provavelmente tarde demais** para a meta de janeiro |

### Faixa 3 — adjacentes ao domínio, encaixe mais fraco

| Veículo | URL | Auditoria | Nota |
|---|---|---|---|
| Government Information Quarterly | <https://www.sciencedirect.com/journal/government-information-quarterly> | 403 (bloqueio a robôs; ativo) | Revista de maior impacto em transparência; ciclo de revisão longo |
| JURIX | <https://jurix.nl/> | **200 OK** | Informática jurídica; a LAI é um problema de prazo legal. Chamada em geral por setembro, logo **edição de 2027, não janeiro** |
| ICAIL | <https://www.iaail.org/> → `iaail.org` | **200 OK** | IA e Direito, bienal — verificar se 2027 é ano de edição |
| IFIP EGOV | [anais na Springer](https://link.springer.com/conference/egov) | **200 OK** | ⚠️ `egov-conference.org` redireciona para `ww38.egov-conference.org`, um **domínio estacionado** — não usar. Usar a âncora da Springer |
| Public Administration Review | <https://onlinelibrary.wiley.com/journal/15406210> | 403 (bloqueio a robôs; ativo) | Exigiria contribuição teórica muito mais pesada |

### Explicitamente rejeitados

| Veículo | URL | Auditoria | Por que não |
|---|---|---|---|
| BRACIS | <https://bracis.sbc.org.br/> | **200 OK** | Veículo de aprendizado de máquina orientado a desempenho; nosso resultado é uma não-melhoria |
| SBBD | <https://sbbd.org.br/> | **200 OK** | Foco em bancos de dados, não é a contribuição |
| SBSI | <https://sbsi.sbc.org.br/> → `/2027/` | **200 OK** | O encaixe em sistemas de informação é plausível, mas mais fraco que a RSP em termos de impacto |
| ECML PKDD | <https://ecmlpkdd.org/> | **200 OK** | A trilha de Applied Data Science espera ganho de desempenho |
| KDMiLe | — | **404 / ERR** em todos os hospedeiros testados | Não foi possível localizar sítio ativo; excluído apenas por isso |

---

## Enquadramento por veículo

O mesmo artefato, três artigos diferentes:

- **FAccT** — *"Vazamento como problema de equidade: auditoria de um modelo de
  triagem administrativa antes da implantação."* Abrir pelos três achados de
  vazamento e pela auditoria de equidade. O resultado negativo passa a ser o
  argumento: a versão implantável é uma tabela transparente, *mais* auditável
  que o GBDT.
- **RSP / RAP** — *"Triagem de pedidos LAI: o que os dados do Fala.BR permitem
  (e não permitem) prever."* Abrir pela consequência operacional:
  `Escolaridade` está 76% ausente, logo a regra de abstenção do Termo de
  Abertura recusaria **77,58%** dos pedidos. Acionável para a CGU.
- **dg.o / Data & Policy** — *"Viabilidade, no instante da chegada, de triagem
  preditiva numa plataforma nacional de acesso à informação."* Abrir pelo
  protocolo; posicionar como reutilizável em qualquer tarefa de predição sobre
  dados abertos governamentais.

---

## Cronograma até janeiro de 2027

Contando de trás para frente a partir de um prazo no fim de janeiro, com hoje em
16/09/2026:

| Janela | Entrega | Situação |
|---|---|---|
| set/2026 | Auditoria de vazamento, linha de base honesta, auditoria de equidade | **concluído** — `docs/VERIFICATION.md` |
| **até 30/09/2026** | **Confirmar datas do FAccT 2027 e do dg.o 2027 nas chamadas oficiais** | **bloqueante — fazer primeiro** |
| out/2026 | Passagem por Scopus/WoS para endurecer a alegação de "nenhum trabalho anterior"; acrescentar 2012–2021 para série de uma década | não iniciado |
| out/2026 | Calibração e simulação de capacidade (quanto uma fila real de analistas seniores absorve) | não iniciado |
| nov/2026 | Auditoria de equidade completa nos 24% de linhas que **têm** dado de perfil; cortes interseccionais | não iniciado |
| nov/2026 | Rascunho v1 | — |
| dez/2026 | Revisão interna; leitura por praticantes da CGU/SIC para a versão da RSP | — |
| jan/2027 | Submeter ao FAccT; submeter à RSP em paralelo (fluxo contínuo) | — |

**Maior risco isolado:** as datas do FAccT 2027 não estavam publicadas na data
da auditoria. Se o prazo cair antes do fim de janeiro, o trabalho de outubro e
novembro comprime. A submissão paralela à RSP existe precisamente para que a
meta de janeiro seja cumprida de todo modo.

**Segundo risco:** a alegação de "nenhum trabalho anterior" repousa hoje em
OpenAlex e Crossref, com o Semantic Scholar indisponível (HTTP 429). Revisores
do FAccT vão sondar isso. A passagem por Scopus/WoS em outubro não é opcional.

---

## Âncoras de dados e de legislação (auditadas)

| Recurso | URL | Auditoria |
|---|---|---|
| CGU Dados Abertos — downloads do Fala.BR | <https://dadosabertos-download.cgu.gov.br/FalaBR/> | **200 OK** |
| Plataforma Fala.BR | <https://falabr.cgu.gov.br/> → `/web/home` | **200 OK** |
| Portal Brasileiro de Dados Abertos | <https://dados.gov.br/> | **200 OK** |
| Lei 12.527/2011 (LAI) | <https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2011/lei/l12527.htm> | **200 OK** |
| API do OpenAlex (procedência da varredura) | <https://api.openalex.org/> | **200 OK** |
| SBC OpenLib (anais brasileiros) | <https://sol.sbc.org.br/> → `/index.php/indice` | **200 OK** |
