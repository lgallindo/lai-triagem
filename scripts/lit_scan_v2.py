"""Varredura dirigida: filtrada por título e resumo (não pelo `search` difuso), para
que seja possível distinguir um conjunto de resultados vazio de um ruidoso.
"""

import json
import time
import urllib.parse
import urllib.request

MAILTO = "lgms@cesar.org.br"
UA = {"User-Agent": f"lai-triagem-litscan (mailto:{MAILTO})"}

# Filtros precisos de título e resumo. Resultado vazio aqui é evidência real.
PROBES = [
    ("A. Fala.BR / LAI specific", [
        "Fala.BR", "Lei de Acesso a Informacao", "acesso a informacao pedidos",
    ]),
    ("B. FOIA / public-records + ML", [
        "freedom of information requests machine learning",
        "FOIA machine learning", "public records requests prediction",
        "information request classification agency",
    ]),
    ("C. Citizen-request routing (311 / ouvidoria)", [
        "311 service requests classification", "citizen complaints classification routing",
        "ouvidoria classificacao automatica", "service request routing machine learning government",
    ]),
    ("D. Public-sector decision support & fairness", [
        "algorithmic decision support public administration",
        "automation bias public sector algorithmic recommendations",
        "fairness public sector risk scoring",
    ]),
    ("E. Misrouting / rework in administrative workflows", [
        "misrouting administrative requests prediction",
        "case routing prediction workflow government",
    ]),
]


def oa(filter_q: str, n: int = 6):
    p = urllib.parse.urlencode({
        "filter": f"title_and_abstract.search:{filter_q}",
        "per-page": n, "mailto": MAILTO,
        "select": "title,publication_year,cited_by_count,doi,primary_location",
        "sort": "cited_by_count:desc",
    })
    try:
        req = urllib.request.Request(f"https://api.openalex.org/works?{p}", headers=UA)
        with urllib.request.urlopen(req, timeout=45) as r:
            d = json.loads(r.read().decode("utf-8"))
        return d.get("meta", {}).get("count", 0), d.get("results", [])
    except Exception as e:
        return -1, [f"error: {e}"]


for group, probes in PROBES:
    print(f"\n{'#' * 78}\n{group}\n{'#' * 78}")
    for probe in probes:
        total, res = oa(probe)
        print(f"\n  >> \"{probe}\"   -> {total} works in OpenAlex")
        if total == 0:
            print("     (no matches — gap indicator)")
        for w in res:
            if isinstance(w, str):
                print(f"     {w}")
                continue
            loc = (w.get("primary_location") or {}).get("source") or {}
            print(f"     [{w.get('publication_year')}] cit={w.get('cited_by_count'):<5} "
                  f"{(w.get('title') or '')[:96]}")
            print(f"            {(loc.get('display_name') or 'n/a')[:86]}")
        time.sleep(1.1)
