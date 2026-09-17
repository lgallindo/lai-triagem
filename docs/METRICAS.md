# Métricas — GERADO AUTOMATICAMENTE, NÃO EDITAR À MÃO

Gerado por `scripts/train.py` em 2026-09-17T21:04:25Z.
Qualquer número de desempenho citado em outro documento deve vir daqui.

## Artefato

| Item | Valor |
|---|---|
| Variáveis | **22** |
| Árvores | **56** |
| Limiar (fila de 10%) | **0.15509** |
| Retrato dos dados | `20260914` |
| Anos de treino | [2022, 2023, 2024] |
| Maturação | 60 dias |
| Campos excluídos por vazamento | 14 |
| Campos enviados pelo chamador | 8 (atômico) |

## Teste 2026 maturado — modelo contra linha de base

| Escore | ROC-AUC | PR-AUC | prec@1% | prec@5% | prec@10% |
|---|---|---|---|---|---|
| Consulta por órgão (sem modelo) | 0.7434 | 0.1641 | 35.67% | 24.80% | 17.86% |
| LightGBM | 0.7662 | 0.1853 | 36.47% | 24.34% | 19.76% |
| Diferença relativa | — | +12.9% | +2.2% | **-1.8%** | +10.6% |

## Custo dos vazamentos (variantes diagnósticas, nunca implantadas)

| Variante | PR-AUC teste maturado | vs honesta |
|---|---|---|
| `with_assunto` | 0.1637 | -2.16 pp |
| `LEAKY_with_prazo` | 0.4022 | +21.69 pp |

## Ressalvas que não se leem nos números

- A precisão@k é o **valor esperado** sob desempate uniforme. A linha de
  base tem blocos grandes de empate, porque a taxa por órgão é constante
  dentro do órgão.
- As variáveis de desfecho usam defasagem de **60 dias**;
  sem ela consumiriam resultados que em produção não seriam conhecidos.
- As variáveis demográficas vêm de um **retrato atual** do cadastro, não do
  perfil na abertura do pedido (H6). Ver `VERIFICATION.md`.
