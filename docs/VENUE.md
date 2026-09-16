# Publishing venue proposal — submit by January 2027

**Link audit date: 2026-09-16.** Every URL below was fetched
(`audit_links.py`, `audit_links2.py`); status recorded verbatim. `403` means the
publisher blocks automated clients — the page is live in a browser, and the DOI
was separately verified through Crossref. **Deadlines are not audited** and must
be confirmed against each official CFP; 2027 calls were largely unpublished at
audit time.

---

## Executive summary

The publishable asset is **not the model** — it is the **leakage audit** and the
**negative result**. Three of four candidate feature families proved post-hoc or
unusable, and after removing them a one-line `groupby` matches a tuned LightGBM
(precision@5% 24.79% vs 24.44% on matured 2026 data). That is a methods-and-
cautionary-tale paper about machine learning on administrative open data, not a
performance paper.

This shapes venue choice decisively. Performance-oriented ML venues have no
place for "our model did not beat a lookup table". Venues that reward
methodological rigour, negative results, and public-sector deployment realism do.

**Recommended: two submissions from one artifact.**

1. **ACM FAccT 2027** — primary. Algorithmic decision support in government,
   with an equity audit (`Ensino Fundamental` flagged at 2.40× population share)
   and a documented refusal to ship a leaky model. FAccT explicitly values
   negative and critical results. Historically a **late-January** abstract
   deadline, which matches the constraint exactly but is also the main risk.
2. **Revista do Serviço Público (ENAP)** — parallel, Portuguese, rolling
   submission, no deadline risk. Reaches the audience that can act on it: CGU and
   federal SIC practitioners. A rolling journal de-risks the FAccT timing gamble.

**Fallback if FAccT timing slips:** `dg.o 2027` (ACM Digital Government
Research), same community, typically a slightly later call.

**Do not target:** BRACIS / SBBD / KDMiLe / ECML PKDD. All reward predictive
performance, which this artifact deliberately does not deliver.

---

## Ranked shortlist

### Tier 1 — recommended

| Venue | URL | Audit | Fit |
|---|---|---|---|
| **ACM FAccT 2027** | <https://facctconference.org/> | **200 OK** | Public-sector algorithmic scoring, equity audit, negative results in scope. Deadline historically late Jan — **confirm first** |
| **Revista do Serviço Público (ENAP)** | <https://revista.enap.gov.br/index.php/RSP> | **200 OK** | PT-BR, rolling, practitioner audience is CGU/SIC itself. No deadline risk |

### Tier 2 — strong alternates

| Venue | URL | Audit | Fit |
|---|---|---|---|
| **dg.o 2027** (ACM Digital Government) | <https://dgsociety.org/> · [dg.o 2026](https://dgsociety.org/dgo-2026/) | **200 OK** (both) | Best pure e-government fit; ACM proceedings. Use as FAccT fallback |
| **Data & Policy** (Cambridge) | <https://www.cambridge.org/core/journals/data-and-policy> | 429 (rate-limited, live) | Open access, policy-facing, rolling. Explicitly publishes negative/implementation findings |
| **Revista de Administração Pública** (FGV) | <https://periodicos.fgv.br/rap> | **200 OK** | Top Brazilian public-administration journal, rolling |
| **AIES 2027** (AAAI/ACM AI Ethics & Society) | <https://www.aies-conference.com/> → `/2026/` | **200 OK** | Ethics framing fits; deadline usually ~March, **likely too late** for Jan target |

### Tier 3 — domain-adjacent, weaker fit

| Venue | URL | Audit | Note |
|---|---|---|---|
| Government Information Quarterly | <https://www.sciencedirect.com/journal/government-information-quarterly> | 403 (bot-blocked, live) | Highest-impact transparency journal; long review cycle |
| JURIX | <https://jurix.nl/> | **200 OK** | Legal informatics; LAI is a legal-deadline problem. Call usually ~Sept, so **2027 edition, not Jan** |
| ICAIL | <https://www.iaail.org/> → `iaail.org` | **200 OK** | AI & Law, biennial — check whether 2027 is an edition year |
| IFIP EGOV | [Springer proceedings](https://link.springer.com/conference/egov) | **200 OK** | ⚠️ `egov-conference.org` redirects to `ww38.egov-conference.org`, a **parked domain** — do not use it. Use the Springer anchor |
| Public Administration Review | <https://onlinelibrary.wiley.com/journal/15406210> | 403 (bot-blocked, live) | Would need a much heavier theory contribution |

### Explicitly rejected

| Venue | URL | Audit | Why not |
|---|---|---|---|
| BRACIS | <https://bracis.sbc.org.br/> | **200 OK** | Performance-driven ML venue; our result is a non-improvement |
| SBBD | <https://sbbd.org.br/> | **200 OK** | Database focus, not the contribution |
| SBSI | <https://sbsi.sbc.org.br/> → `/2027/` | **200 OK** | Information-systems fit is plausible but weaker than RSP for impact |
| ECML PKDD | <https://ecmlpkdd.org/> | **200 OK** | Applied Data Science track expects a performance win |
| KDMiLe | — | **404 / ERR** on every candidate host | Could not locate a live site; excluded on that basis alone |

---

## Framing per venue

The same artifact, three different papers:

- **FAccT** — *"Leakage as a fairness problem: auditing an administrative
  triage model before deployment."* Lead with the three leakage findings and the
  equity audit. The negative result becomes the argument: the deployable version
  is a transparent lookup table, which is *more* auditable than the GBDT.
- **RSP / RAP** — *"Triagem de pedidos LAI: o que os dados do Fala.BR permitem
  (e não permitem) prever."* Lead with operational consequence: `Escolaridade` is
  76% missing, so the charter's abstention rule would decline **77.58%** of
  requests. Actionable for CGU.
- **dg.o / Data & Policy** — *"Arrival-time feasibility of predictive triage in
  a national FOI platform."* Lead with the protocol; position as reusable for any
  government open-data prediction task.

---

## Timeline to January 2027

Working back from a late-January deadline, with today at 2026-09-16:

| Window | Deliverable | Status |
|---|---|---|
| Sep 2026 | Leakage audit, honest baseline, equity audit | **done** — `docs/VERIFICATION.md` |
| **by 2026-09-30** | **Confirm FAccT 2027 + dg.o 2027 dates from official CFPs** | **blocking — do first** |
| Oct 2026 | Scopus/WoS pass to harden the "no prior work" claim; add 2012–2021 years for a decade-long trend | not started |
| Oct 2026 | Calibration + capacity simulation (what a real senior-analyst queue can absorb) | not started |
| Nov 2026 | Full equity audit on the 24% of rows that *have* profile data; intersectional slices | not started |
| Nov 2026 | Draft v1 | — |
| Dec 2026 | Internal review; CGU/SIC practitioner read for the RSP version | — |
| Jan 2027 | Submit FAccT; submit RSP in parallel (rolling) | — |

**Single largest risk:** FAccT 2027 dates were unpublished at audit time. If the
deadline lands before late January, the Oct–Nov work compresses. The RSP
parallel submission exists precisely so the January target is met regardless.

**Second risk:** the "no prior work" claim currently rests on OpenAlex +
Crossref, with Semantic Scholar unavailable (HTTP 429). Reviewers at FAccT will
probe this. The October Scopus/WoS pass is not optional.

---

## Data and legal anchors (audited)

| Resource | URL | Audit |
|---|---|---|
| CGU Fala.BR open-data downloads | <https://dadosabertos-download.cgu.gov.br/FalaBR/> | **200 OK** |
| Fala.BR platform | <https://falabr.cgu.gov.br/> → `/web/home` | **200 OK** |
| Portal Brasileiro de Dados Abertos | <https://dados.gov.br/> | **200 OK** |
| Lei 12.527/2011 (LAI) | <https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2011/lei/l12527.htm> | **200 OK** |
| OpenAlex API (scan provenance) | <https://api.openalex.org/> | **200 OK** |
| SBC OpenLib (BR proceedings) | <https://sol.sbc.org.br/> → `/index.php/indice` | **200 OK** |
