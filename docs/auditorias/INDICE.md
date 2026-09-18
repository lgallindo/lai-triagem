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

## O que estas auditorias têm em comum

Todas foram desenhadas para **encontrar defeito**, não para atestar qualidade, e
todas encontraram. Os três defeitos graves da primeira eram do autor do código.
Os dois graves da terceira também. Um projeto cuja conclusão é "o modelo não
supera uma tabela de consulta de uma linha" só tem valor se o procedimento que
chegou a ela for hostil ao próprio resultado.

## O que ainda não foi auditado

- **Dimensão 7 por fornecedor terceiro**: a auditoria de clareza foi feita por
  um subagente do mesmo modelo que escreveu o código. Restrição de contexto não
  é o mesmo que independência de fornecedor.
- **Prosa, artefato entregue e comentários de código**: camadas acrescentadas
  depois de 18/09/2026, quando ficou medido que uma guarda numérica não vê
  contradição em prosa.
