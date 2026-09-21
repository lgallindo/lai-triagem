"""Varredura da literatura via OpenAlex e Crossref, em busca de trabalho anterior
sobre tratamento assistido por aprendizado de máquina de pedidos de acesso à
informação.
"""

import json
import time
import urllib.parse
import urllib.request

MAILTO = "lgms@cesar.org.br"
UA = {"User-Agent": f"lai-triagem-litscan (mailto:{MAILTO})"}

QUERIES = [
    "freedom of information request classification machine learning",
    "public records request triage prediction government",
    "FOIA requests text classification prediction",
    "Lei de Acesso a Informacao aprendizado de maquina pedidos",
    "transparency portal request routing classification e-government",
    "administrative burden prediction public service requests machine learning",
    "citizen request classification municipal 311 machine learning routing",
    "ombudsman complaint classification machine learning public sector",
    "risk scoring prioritization public administration case triage",
    "algorithmic decision support public sector fairness education level",
]


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode("utf-8"))


def openalex(q: str, n: int = 8):
    p = urllib.parse.urlencode({
        "search": q, "per-page": n, "mailto": MAILTO,
        "select": "title,publication_year,cited_by_count,doi,primary_location,type",
    })
    try:
        return get(f"https://api.openalex.org/works?{p}").get("results", [])
    except Exception as e:
        print(f"    !! openalex failed: {e}")
        return []


def venue_of(w):
    loc = (w.get("primary_location") or {}).get("source") or {}
    return loc.get("display_name") or "n/a"


seen = {}
for q in QUERIES:
    print(f"\n{'=' * 78}\nQUERY: {q}\n{'=' * 78}")
    for w in openalex(q):
        title = (w.get("title") or "").strip()
        if not title:
            continue
        key = title.lower()[:70]
        yr = w.get("publication_year")
        cit = w.get("cited_by_count", 0)
        print(f"  [{yr}] cit={cit:<5} {title[:112]}")
        print(f"         venue: {venue_of(w)[:95]}")
        if w.get("doi"):
            print(f"         doi:   {w['doi']}")
        if key not in seen or cit > seen[key][1]:
            seen[key] = (title, cit, yr, venue_of(w), w.get("doi"))
    time.sleep(1.1)

print(f"\n\n{'#' * 78}\nDEDUPED, RANKED BY CITATIONS ({len(seen)} unique works)\n{'#' * 78}")
for title, cit, yr, venue, _doi in sorted(seen.values(), key=lambda t: -t[1])[:35]:
    print(f"[{yr}] cit={cit:<6} {title[:100]}")
    print(f"        {venue[:88]}")
