"""Sondagens específicas de LAI e FOIA. Contagem zero aqui é a evidência desejada.
"""

import json
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "lai-litscan (mailto:lgms@cesar.org.br)"}
PROBES = [
    "Fala.BR",
    "Lei de Acesso a Informacao",
    "access to information requests Brazil",
    "reencaminhamento pedidos acesso informacao",
    "freedom of information requests machine learning",
    "FOIA request prediction",
    "transparency requests machine learning Brazil",
    "public information request triage prioritization",
    "government information request misrouting",
    "forwarding prediction public request",
]


def oa(q, n=6):
    p = urllib.parse.urlencode({
        "filter": f"title_and_abstract.search:{q}",
        "per-page": n,
        "mailto": "lgms@cesar.org.br",
        "select": "title,publication_year,cited_by_count,doi,primary_location",
        "sort": "cited_by_count:desc",
    })
    try:
        req = urllib.request.Request(f"https://api.openalex.org/works?{p}", headers=UA)
        with urllib.request.urlopen(req, timeout=45) as f:
            d = json.loads(f.read().decode())
        return d.get("meta", {}).get("count", 0), d.get("results", [])
    except Exception as e:
        print(f"   !! {e}")
        return -1, []


for q in PROBES:
    total, res = oa(q)
    print(f'\n>> "{q}" -> {total} works')
    if total == 0:
        print("   (ZERO — gap indicator)")
    for w in res:
        loc = (w.get("primary_location") or {}).get("source") or {}
        year = w.get("publication_year")
        cit = w.get("cited_by_count")
        title = (w.get("title") or "")[:90]
        venue = (loc.get("display_name") or "n/a")[:80]
        print(f"   [{year}] cit={cit} {title}")
        print(f"        {venue}")
    time.sleep(1.1)
