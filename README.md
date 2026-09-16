# lai-triagem — reencaminhamento-risk triage for Brazilian LAI requests

Flags an incoming freedom-of-information request on the federal **Fala.BR**
platform as `ALTO RISCO` / `BAIXO RISCO` of internal re-routing
(`FoiReencaminhado`), so senior SIC analysts can be pointed at the requests most
likely to be misrouted.

> **Read [`docs/VERIFICATION.md`](docs/VERIFICATION.md) before using the score.**
> With leakage removed, a one-line organ-rate lookup matches the trained model.
> Both are exposed by the service so you can compare them.

## Headline numbers

Matured 2026 test set (registered ≥60 days before the 2026-09-14 snapshot),
trained on 2022–2024, validated on 2025. Base rate 5.75%.

| Scorer | ROC-AUC | PR-AUC | prec@1% | prec@5% | prec@10% |
|---|---|---|---|---|---|
| Organ-rate lookup (no model) | 0.7434 | 0.1641 | 36.12% | **24.79%** | 17.79% |
| LightGBM, 161 trees | 0.7471 | **0.1723** | **36.59%** | 24.44% | **18.70%** |

**4.3× lift at the top-5% queue** — operationally useful, but the model's
advantage over the lookup is within noise, and it loses at precision@5%.

## Excluded features, and why

Each exclusion is empirical, not precautionary — see the audit.

| Field | Reason |
|---|---|
| `FoiProrrogado`, `Situacao`, `DataResposta`, `Decisao`, `EspecificacaoDecisao`, `DetalhamentoDecisao`, `MotivoNegativaAcesso`, `PrazoRestricaoAcesso` | populated after the response |
| `AssuntoPedido`, `SubAssuntoPedido`, `Tag` | assigned during triage — 79.6% missing at 3 days old, 0.000% once answered |
| `PrazoAtendimento` / `prazo_dias` | rewritten on prorrogation (+10 d, LAI art. 11 §2). Held 40.5% of gain and inflated PR-AUC by **+22 pp** |

The service **refuses to score** a payload containing any of them.

## Quick start (WSL)

```bash
cd ~/lai-triagem
uv venv --python 3.12 && uv pip install pandas pyarrow lightgbm scikit-learn bentoml
```

Fetch data (~38 MB, 5 years, no request text):

```bash
for y in 2022 2023 2024 2025 2026; do curl -sS -o data/raw/Pedidos_csv_$y.zip "https://dadosabertos-download.cgu.gov.br/FalaBR/Arquivos_FalaBR/Pedidos_csv_$y.zip"; done
```

Train — **~20 s wall clock**, 2 s per fit:

```bash
.venv/bin/python scripts/train.py
```

Score one request, no BentoML needed:

```bash
.venv/bin/python examples/minimal_predict.py
```

```
organ            : CC-PR – Casa Civil da Presidência da República
historical rate  : 0.4786   (cohort base 0.0803)
P(reencaminhado) : 0.4691
flag             : ALTO RISCO   (threshold 0.1691)
contrast, UFLA   : 0.0057  ->  BAIXO RISCO
leakage guard    : OK — post-hoc field(s) supplied, refusing to score: ['FoiProrrogado']
```

## BentoML

The artifact is a LightGBM **native text** model plus a JSON sidecar — no
pickles, so no version coupling on load.

```bash
.venv/bin/python scripts/register_bento.py      # -> BentoML model store
.venv/bin/bentoml serve service.py:LaiTriagem
```

Endpoints: `/score` (model), `/score_baseline` (organ lookup, for comparison),
`/health` (tag, tree count, feature list, exclusions, snapshot).

## Layout

| Path | Role |
|---|---|
| `scripts/train.py` | temporal-split training, both variants + the leaky diagnostic + baseline |
| `scripts/verify_leakage_v2.py` | H1 `AssuntoPedido`, H2 duplicate/Recursos tests |
| `scripts/verify_h2_final.py`, `verify_h2_baserate.py` | H2 directional test |
| `scripts/verify_h3_prazo.py` | H3 deadline leakage, H4 demographic availability |
| `scripts/lit_scan*.py` | literature scan (OpenAlex/Crossref) |
| `lai_triagem/featurize.py` | arrival-time featurisation + leakage guard, shared by train and serve |
| `service.py`, `scripts/register_bento.py` | BentoML service and model registration |
| `examples/minimal_predict.py` | minimal scoring example |
| `docs/VERIFICATION.md` | the leakage/feasibility audit — **start here** |
| `docs/LITERATURE.md` | scholarship scan + protocol |
| `docs/VENUE.md` | publication plan, audited links |

## Known limitations

- **`Escolaridade` is 76.2% missing**, `Profissao` 76.9%. The charter's
  abstention-on-missing-profile rule would decline **77.58%** of requests.
- Demographics contribute ~5% of model gain; 76% is organ identity.
- 2026 labels are right-censored (positive rate 5.75% matured vs 3.60% newer).
- `orgao_rate` is a static lookup fitted on train years; it needs periodic refit.
- Threshold 0.1691 is a top-10%-queue operating point, not a calibrated probability.

## Data and licence

Source: CGU Dados Abertos,
<https://dadosabertos-download.cgu.gov.br/FalaBR/Arquivos_FalaBR/> — snapshot
`20260914`. Uses the no-text archives (~7–9 MB/year); request text is out of
scope by design. Code MIT. No data is committed: plain git only, no DVC, no LFS.
