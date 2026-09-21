# Métricas — GERADO AUTOMATICAMENTE, NÃO EDITAR À MÃO

Gerado por `scripts/train.py` em 2026-09-21T14:54:50Z.
Qualquer número de desempenho citado em outro documento deve vir daqui.

## Artefato

| Item | Valor |
|---|---|
| Variáveis | **31** |
| Árvores | **56** |
| Limiar (fila de 10%) | **0.159391** |
| Retrato dos dados | `20260914` |
| Anos de treino | [2022, 2023, 2024] |
| Maturação | 60 dias |
| Campos excluídos por vazamento | 14 |
| Campos enviados pelo chamador | 8 (atômico) |

## Teste 2026 maturado — modelo contra linha de base

| Escore | ROC-AUC | PR-AUC | prec@1% | prec@5% | prec@10% |
|---|---|---|---|---|---|
| Consulta por órgão (sem modelo) | 0.7434 | 0.1641 | 35.67% | 24.80% | 17.86% |
| LightGBM | 0.7726 | 0.1901 | 36.47% | 24.23% | 19.61% |
| Diferença relativa | — | +15.8% | +2.2% | **-2.3%** | +9.7% |

## Custo dos vazamentos (variantes diagnósticas, nunca implantadas)

| Variante | PR-AUC teste maturado | vs honesta |
|---|---|---|
| `with_assunto` | 0.1613 | -2.88 pp |
| `LEAKY_with_prazo` | 0.4094 | +21.93 pp |

## Ressalvas que não se leem nos números

- A precisão@k é o **valor esperado** sob desempate uniforme. A linha de
  base tem blocos grandes de empate, porque a taxa por órgão é constante
  dentro do órgão.
- As variáveis de desfecho usam defasagem de **60 dias**;
  sem ela consumiriam resultados que em produção não seriam conhecidos.
- As variáveis demográficas vêm de um **retrato atual** do cadastro, não do
  perfil na abertura do pedido (H6). Ver `VERIFICATION.md`.
