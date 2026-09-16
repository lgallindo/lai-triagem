# Leakage and feasibility audit — LAI reencaminhamento risk

Every claim here is reproducible from `scripts/` against the CGU Fala.BR open
data snapshot **20260914**. Data source:
<https://dadosabertos-download.cgu.gov.br/FalaBR/Arquivos_FalaBR/>

Cohort: `Pedidos_csv_{2022..2026}`, **655,177 rows**, base reencaminhamento rate
**7.32%**. Rows with `Situacao == "Encaminhada por Outro Órgão"` (459) are
dropped from modelling: they are in transit, so their `OrgaoDestinatario` is the
receiver rather than the addressee.

## Summary of findings

| ID | Hypothesis | Verdict | Script |
|----|-----------|---------|--------|
| H1 | `AssuntoPedido` is assigned during triage, not at intake | **CONFIRMED — excluded** | `verify_leakage_v2.py` |
| H2 | `OrgaoDestinatario` is overwritten on forwarding | **REFUTED — retained** | `verify_h2_final.py`, `verify_h2_baserate.py` |
| H3 | `prazo_dias` is an arrival-time feature | **REFUTED — it is leaky, excluded** | `verify_h3_prazo.py` |
| H4 | Requester demographics are usable | **LARGELY UNAVAILABLE** | `verify_h3_prazo.py` |

## H1 — `AssuntoPedido` is a triage output

Measured on the 2026 file, whose recent rows are genuinely untriaged (snapshot
date equals the newest `DataRegistro`). The 2024 file shows only 0.098% missing
and hides the effect entirely, because every 2024 row has been triaged for ~2 years.

| Window before snapshot | Rows | `AssuntoPedido` missing |
|---|---|---|
| last 3 d | 902 | **79.60%** |
| last 7 d | 3,189 | 57.98% |
| last 14 d | 6,018 | 49.47% |
| last 30 d | 13,811 | 32.46% |
| last 90 d | 41,299 | 12.41% |
| last 365 d | 113,538 | 4.64% |

By status: `Cadastrada` **59.5%** missing vs `Concluída` **0.000%**.
By response: unanswered **58.2%** vs answered **0.000%**.

A monotone decay to zero as requests age is the signature of a backfilled field.
**Consequence:** a model trained on closed historical requests sees ~100%
coverage and would meet ~60% NULLs in production — train/serve skew.

## H2 — `OrgaoDestinatario` is the addressed organ

Three independent tests:

1. **No duplicate records.** 655,177 rows carry 655,177 distinct `IdPedido`
   *and* 655,177 distinct `ProtocoloPedido`. Forwarding never creates a second row.
2. **`Recursos` join.** `Pedidos.OrgaoDestinatario == Recursos.OrgaoPedido` at
   **100.000%** for both reencaminhado and non-reencaminhado rows (2024: n=10,089;
   2025: n=11,919). This proves internal consistency, *not* direction — both
   files come from the same snapshot.
3. **Directional test (decisive).** Under an overwrite, organs that shed
   misaddressed requests would show *low* rates and absorbers *high* rates.
   Observed is the opposite:

| Organ | Pedidos | Reenc. rate |
|---|---|---|
| SGPR – Secretaria-Geral da Presidência | 1,645 | **50.58%** |
| GSI-PR – Gabinete de Segurança Institucional | 1,652 | 44.25% |
| CC-PR – Casa Civil | 6,086 | 42.56% |
| MGI – Ministério da Gestão | 11,940 | 22.79% |
| *90 federal universities (mean)* | — | **1.09%** |

Central/Presidency mean **36.69%** vs narrow-scope mean **1.09%** — a **33.7×**
spread. Casa Civil cannot plausibly be the leading *destination* of forwarded
requests; it is where citizens send what they cannot place. This matches the
project charter's own premise.

Rejected test, recorded for completeness: the NUP protocol prefix
(`ProtocoloPedido[:5]`) is **not** an organ identifier — weighted purity 0.58.

## H3 — `prazo_dias` is leaky (the important one)

`prazo_dias = PrazoAtendimento − DataRegistro` was the trained model's dominant
feature at **40.5% of gain**. It is post-hoc:

| `FoiProrrogado` | n | mean | median | p05 | p95 |
|---|---|---|---|---|---|
| Não | 518,948 | 21.59 | **21.0** | 20.0 | 25.0 |
| Sim | 136,228 | 31.29 | **31.0** | 21.0 | 39.0 |

The **+10 day** median difference is exactly the single extension granted by
[Lei 12.527/2011](https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2011/lei/l12527.htm)
art. 11 §2. `PrazoAtendimento` is rewritten when the extension is granted —
after intake. So `prazo_dias` re-imported `FoiProrrogado`, which was already on
the exclusion list, through the back door.

Cost of the leak, measured by training a diagnostic variant that keeps it:

| Split | PR-AUC honest | PR-AUC leaky | Inflation |
|---|---|---|---|
| validation 2025 | 0.2053 | 0.4318 | **+22.65 pp** |
| test 2026 | 0.1602 | 0.3772 | **+21.70 pp** |
| test 2026 matured | 0.1723 | 0.3965 | **+22.42 pp** |

Precision@5% on matured test: **24.44% honest vs 43.25% leaky.** The leak nearly
doubles apparent performance.

## H4 — demographics are mostly absent

| Field | Missing | Distinct |
|---|---|---|
| `Escolaridade` | **76.23%** | 6 |
| `Profissao` | **76.87%** | 15 |
| `Genero` | 70.67% | 3 |
| `TipoDemandante` | 16.91% | 2 |

`IdSolicitante == '0'` (anonymised) on 110,744 rows (16.90%).

The charter's abstention rule — decline to score when profile data is missing —
would fire on **77.58%** of all requests, which is not a viable triage product.
The missingness is also **not informative**: reencaminhamento rate is 7.54% when
the profile is present vs 7.25% when absent.

## Right-censoring

The 2026 file is a 2026-09-14 snapshot, so a September request has had days, not
months, to be forwarded. Confirmed: positive rate **5.75%** on rows registered
≥60 days before the snapshot vs **3.60%** on newer rows. Metrics are therefore
reported on both the full and the matured test set (`MATURITY_DAYS = 60`).

## The headline result: ML barely beats a lookup table

Honest arrival-time features only. Baseline = rank by smoothed historical
reencaminhamento rate per organ, fitted on train years only — one `groupby`.

| Matured test 2026 | ROC-AUC | PR-AUC | prec@1% | prec@5% | prec@10% |
|---|---|---|---|---|---|
| Organ-rate lookup | 0.7434 | 0.1641 | 36.12% | **24.79%** | 17.79% |
| LightGBM (161 trees) | 0.7471 | 0.1723 | 36.59% | 24.44% | **18.70%** |

The model wins on PR-AUC by 0.008 and on precision@10% by 0.9 pp, and **loses at
precision@5%**. 76% of its gain is organ identity (`orgao_rate` 48.1% +
`OrgaoDestinatario` 27.7%).

**Conclusion:** with leakage removed, essentially all recoverable signal is
"some organs are chronically misaddressed". That is still operationally useful —
**4.3× lift at the top-5% queue** — but it does not require machine learning,
and any published claim must say so.
