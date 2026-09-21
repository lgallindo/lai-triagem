# Auditorias — índice

Cada auditoria tem arquivo próprio, nomeado pela data em que foi feita. Os
números dentro de cada arquivo são os **daquele momento**: não os leia como
correntes. Para os correntes, sempre [`../METRICAS.md`](../METRICAS.md), que é
gerado pelo treinamento.

| Data | Auditor | Escopo | Achados |
|---|---|---|---|
| [17/09/2026](2026-09-17-codigo-codex.md) | `codex` (fornecedor externo) | código e dados: vazamento, causalidade, métricas, contrato, reprodução | 7, dos quais **3 GRAVES** |
| [17/09/2026](2026-09-17-clareza-subagente.md) | subagente de contexto restrito | o README é seguível por um iniciante? | 9; 3 de 4 objetivos cumpridos |
| [18/09/2026](2026-09-18-camadas-0-1.md) | o próprio autor, roteiro fixado antes | camadas 0 e 1: guardas e reprodutibilidade | 6, dos quais **2 GRAVES** |
| [18/09/2026](2026-09-18-camadas-2-3.md) | `codex` (`gpt-5.6-sol`), fornecedor terceiro | camada 3: o README é seguível por iniciante? camada 2: código e dados | 10, dos quais **2 GRAVES** — um deles mudou o resultado principal |
| [21/09/2026](2026-09-21-boas-praticas.md) | o próprio autor, com `ruff` e `mypy` | adequação a boas práticas antes de comparar bibliotecas | 1 defeito **vivo** (H9 corrigido em 1 de 6 cópias), 1 achado pelo mypy, e o **H10**, maior ganho de todo o processo |
| [21/09/2026](2026-09-21-featuretools.md) | o próprio autor, roteiro fixado antes | ramo `experimento/featuretools`: o `cutoff_time` cumpre a promessa de impedir vazamento por construção? | resultado reproduzido na casa decimal; **nenhum** dos 3 defeitos graves fica fora de alcance; o corte inclui o próprio rótulo por padrão, `training_window` devolve **zero em silêncio** sem `add_last_time_indexes()`, e a construção ficou **978× mais lenta** |

## O que estas auditorias têm em comum

Todas foram desenhadas para **encontrar defeito**, não para atestar qualidade, e
todas encontraram. Os três defeitos graves da primeira eram do autor do código.
Os dois graves da terceira também. Um projeto cuja conclusão é "o modelo não
supera uma tabela de consulta de uma linha" só tem valor se o procedimento que
chegou a ela for hostil ao próprio resultado.

## O que ainda não foi auditado

- **Camada 2 não terminou.** O auditor da dimensão de código esgotou a cota
  depois de 227 mil tokens, tendo deixado duas hipóteses nomeadas no log —
  ambas confirmadas, e uma delas GRAVE. As dimensões que ele **não** chegou a
  cobrir são: métricas (3), documentação contra código (4) por conta própria,
  reprodutibilidade (6) e a tentativa de enganar as guardas (7).
- **Paridade treino/serviço (B5)**: está verificado que `feature_order` casa
  com o booster em nome e ordem, mas **não** que o `featurize` do serviço
  produza os mesmos *valores* que o treinamento produziu para o mesmo pedido.
  É a lacuna conhecida mais relevante.
- **Colisão de caixa em `artifacts/`**: `model_SEM_demografia_H6.txt` e
  `model_sem_demografia_H6.txt` diferem só em maiúsculas, e por isso o
  repositório **não pode ser clonado em Windows nem em macOS** sem que um
  sobrescreva o outro. Defeito conhecido, deliberadamente não corrigido antes
  das auditorias, e registrado aqui para que quem o encontrar saiba que já era
  sabido.
- **Ganho por variável ainda é mantido à mão** no README, embora agora exista
  guarda que o confere a cada execução.
