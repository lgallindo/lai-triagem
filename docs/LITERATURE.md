# Scholarship scan — ML for freedom-of-information request handling

**Executed** 2026-09-15/16. **Audit date for all links:** 2026-09-16.

## 1. Protocol (what was actually run)

Reproducible via `scripts/lit_scan.py`, `scripts/lit_scan_v2.py`,
`scripts/lit_scan_lai.py`.

| Step | Tool | Detail |
|---|---|---|
| S1 | OpenAlex `search=` | 10 broad queries, 8 results each, deduped by title, ranked by citations |
| S2 | OpenAlex `filter=title_and_abstract.search:` | 5 thematic groups, 15 precise probes — a **zero count here is evidence**, not noise |
| S3 | OpenAlex, LAI-specific | 10 probes: `Fala.BR`, `Lei de Acesso a Informacao`, `FOIA request prediction`, `government information request misrouting`, … |
| S4 | Crossref REST | DOI-level verification of every work cited below |
| S5 | HTTP audit | `audit_links.py` / `audit_links2.py` — status + redirect chain per URL |

Two methodological caveats, stated because they bound the conclusion:

- **Semantic Scholar was unavailable** (HTTP 429 throughout); OpenAlex + Crossref
  only. A Scopus/WoS pass would strengthen the negative claim.
- **OpenAlex relevance ranking is poor for this topic.** `"freedom of
  information request machine learning"` restricted to title+abstract returns
  **68 works**, of which *none* are about FOIA request processing — the top hits
  are thermal facial recognition and GDPR commentary. Counts alone are therefore
  unreliable; each result set was read.

## 2. Principal finding: the direct literature is empty

No published work was found that predicts **misrouting / rework risk** for
freedom-of-information requests, in Brazil or elsewhere.

Zero-result probes (S2/S3), each a gap indicator:

- `ouvidoria classificacao automatica` → **0**
- `misrouting administrative requests prediction` → **0**
- `Fala.BR` → no on-topic work
- `reencaminhamento pedidos acesso informacao` → no on-topic work

## 3. Adjacent clusters (what does exist)

### 3a. Citizen-request / complaint classification — topic routing, not rework risk

The nearest neighbours classify *what a request is about* in order to route it.
None model the **probability that routing will fail**, which is this artifact's target.

| Work | Year | Venue | Note |
|---|---|---|---|
| [Structure of 311 service requests as a signature of urban location](https://doi.org/10.1371/journal.pone.0186314) | 2017 | PLOS ONE | Crossref-verified. 311 requests as an urban signal; descriptive, not predictive of misrouting |
| Intelligent Ombudsman: An AI-Based Approach to Demand Classification | 2026 | Springer LNCS | Closest institutional analogue (ouvidoria demand classification); 0 citations, very recent |
| Fine-Tuning of a LLM for Public Complaint Classification and Routing | 2026 | Iconic Research and Engineering Journals | LLM routing; low-visibility venue |
| CivicFix: Smart Complaint Routing for Urban Solutions | 2025 | IJARCCE | Municipal complaint routing |
| Automated Classification of Arabic client Inquiries for Government Services | 2025 | — | Government inquiry classification, fine-tuning approach |

### 3b. Public-sector algorithmic decision support — the framing literature

Directly relevant to the charter's design (an analyst *sees a flag* and decides):

| Work | Year | Venue | Cites | Why it matters here |
|---|---|---|---|---|
| [Human–AI Interactions in Public Sector Decision Making: "Automation Bias" and "Selective Adherence"](https://doi.org/10.1093/jopart/muac007) | 2022 | JPART | 423 | The core risk of a "priority flag" product: analysts over-adhere to the score |
| [Accountable Artificial Intelligence: Holding Algorithms to Account](https://doi.org/10.1111/puar.13293) | 2020 | Public Administration Review | 601 | Accountability frame for a state-deployed scorer |
| [Fairness and Accountability Design Needs for Algorithmic Support in High-Stakes Public Sector Decision-Making](https://doi.org/10.1145/3173574.3174014) | 2018 | ACM CHI | 464 | Design requirements for exactly this class of tool |

All three Crossref-verified. Publisher pages return HTTP 403 to automated
clients; the DOIs resolve.

**Correction from an earlier draft of this scan:** a paper on GDPR Art. 22
multi-stage profiling was cited against `10.1093/idpl/ipab002`. Crossref shows
that DOI belongs to *"Data-driven measures to mitigate the impact of COVID-19 in
South America"* (International Data Privacy Law, 2021). The citation was wrong
and has been removed rather than guessed at.

### 3c. Methodological analogue — capacity-bounded triage

Emergency-department triage ML is the closest *methodological* precedent for a
precision@k priority queue under fixed reviewer capacity, e.g. Raita et al. 2019
(*Critical Care*) and Hong et al. 2018 (*PLoS ONE*). Cited for method, not domain.

## 4. The gap this artifact occupies

Three contributions, in descending order of novelty:

1. **Target novelty.** Predicting *administrative rework* (`FoiReencaminhado`)
   rather than the routing label. No prior work located.
2. **A leakage-audit protocol for administrative open data.** Three of four
   candidate feature families turned out to be post-hoc or unusable, and the
   decisive test differed each time (recency decay for `AssuntoPedido`; a
   directional organ-rate argument for `OrgaoDestinatario`; a statutory +10-day
   signature for `prazo_dias`). Published administrative-ML papers rarely report
   this, and the `prazo_dias` case shows why: it inflates PR-AUC by **+22 pp**
   and looks like an excellent feature.
3. **A negative result worth publishing.** After removing leakage, a
   one-`groupby` organ-rate lookup matches LightGBM (prec@5% 24.79% vs 24.44%).
   The equity audit also quantifies the charter's predicted "positive
   discrimination": requesters with `Ensino Fundamental` appear in the top-10%
   alert queue at **2.40×** their population share.

Contributions 2 and 3 are venue-independent and do not depend on the model
working — which, given the finding, is the point.
