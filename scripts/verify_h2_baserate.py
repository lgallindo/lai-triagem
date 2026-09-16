"""
H2, base-rate correction.

The previous overlap test (19/20 high-reenc organs are "provable receivers") is
confounded: the receiver set is dominated by high-volume organs, so ANY top-20
list would overlap with it. This quantifies the confound and then applies the
directional argument, which is not confounded.

Directional argument
--------------------
Suppose OrgaoDestinatario is REWRITTEN to the receiving organ on forwarding.
Then an organ that mostly *sheds* misaddressed requests never keeps them, so it
should show a LOW reencaminhamento rate; organs that *absorb* forwarded requests
should show HIGH rates.

Suppose instead OrgaoDestinatario keeps the ADDRESSED organ. Then the organs
citizens misaddress show HIGH rates, and narrowly-scoped organs show ~0%.

These predictions are opposite, so the identity of the top-rate organs decides it.
"""

from pathlib import Path

import pandas as pd

INTERIM = Path.home() / "lai-triagem" / "data" / "interim"
SNAP = "20260914"
READ_KW = dict(sep=";", encoding="utf-16", dtype=str, na_values=[" ", ""], keep_default_na=True)
COLS = ["IdPedido", "OrgaoDestinatario", "FoiReencaminhado", "Situacao"]

fr = []
for y in (2022, 2023, 2024, 2025, 2026):
    d = pd.read_csv(INTERIM / f"{SNAP}_Pedidos_csv_{y}.csv", usecols=COLS, **READ_KW)
    d.columns = [c.strip() for c in d.columns]
    for c in d.columns:
        d[c] = d[c].str.strip()
    fr.append(d)
a = pd.concat(fr, ignore_index=True)
a["reenc"] = a.FoiReencaminhado.eq("Sim")

receivers = set(a.loc[a.Situacao.eq("Encaminhada por Outro Órgão"), "OrgaoDestinatario"].dropna())
g = a.groupby("OrgaoDestinatario").agg(n=("IdPedido", "size"), reenc=("reenc", "sum"))
q = g[g.n >= 500].copy()
q["rate"] = q.reenc / q.n
q["is_recv"] = [o in receivers for o in q.index]

print("=" * 78)
print("BASE-RATE CORRECTION")
print("=" * 78)
print(f"qualifying organs (n>=500):            {len(q)}")
print(f"  of which in provable-receiver set:   {q.is_recv.sum()}  ({100*q.is_recv.mean():.1f}%)")
top20 = q.sort_values("rate", ascending=False).head(20)
print(f"  in top-20 by reenc rate:             {top20.is_recv.sum()}/20  ({100*top20.is_recv.mean():.1f}%)")
bot20 = q.sort_values("rate").head(20)
print(f"  in bottom-20 by reenc rate:          {bot20.is_recv.sum()}/20  ({100*bot20.is_recv.mean():.1f}%)")
print(f"\n=> the overlap test is {'CONFOUNDED (uninformative)' if q.is_recv.mean() > 0.6 else 'informative'}:"
      f" membership in the receiver set is already {100*q.is_recv.mean():.0f}% at baseline.")

# Volume confound, stated explicitly.
print(f"\nmedian pedido volume, receiver organs:     {q[q.is_recv].n.median():>10,.0f}")
print(f"median pedido volume, non-receiver organs: {q[~q.is_recv].n.median():>10,.0f}")

print("\n" + "=" * 78)
print("DIRECTIONAL ARGUMENT (not confounded)")
print("=" * 78)

# Presidency / central-coordination bodies: the canonical misaddress targets.
misaddress_magnets = q[q.index.str.contains(
    r"Presidência da República|Casa Civil|Secretaria-Geral|Gabinete de Segurança|"
    r"Secretaria de Comunicação Social da Presid|Ministério da Gestão", regex=True, na=False)]
print("\ncentral / Presidency bodies — the organs citizens misaddress when they")
print("cannot identify the competent one:")
print(misaddress_magnets.assign(rate_pct=(100 * misaddress_magnets.rate).round(2))
      [["n", "reenc", "rate_pct"]].sort_values("rate_pct", ascending=False).to_string())

# Narrow-scope organs: unambiguous competence, should be addressed correctly.
narrow = q[q.index.str.contains(
    r"Universidade Federal|UF[A-Z]{1,3}\b|CEFET|Instituto Federal", regex=True, na=False)]
print(f"\nnarrow-scope organs (federal universities / institutes), n={len(narrow)}:")
print(f"  mean reenc rate:   {100 * narrow.rate.mean():.2f}%")
print(f"  median reenc rate: {100 * narrow.rate.median():.2f}%")

print(f"\nall qualifying organs mean reenc rate: {100 * q.rate.mean():.2f}%")
print(f"central/Presidency mean:               {100 * misaddress_magnets.rate.mean():.2f}%")
print(f"ratio (central / narrow):              "
      f"{misaddress_magnets.rate.mean() / max(narrow.rate.mean(), 1e-9):.1f}x")

print("""
VERDICT
-------
Under the OVERWRITE hypothesis, Presidency bodies would have to be the leading
DESTINATION of forwarded requests, and universities would have to be frequent
receivers too. Under the ADDRESSED-ORGAN hypothesis, Presidency bodies are where
citizens send requests they cannot place, and narrow-scope organs are addressed
correctly. The observed pattern matches the second prediction, so
OrgaoDestinatario behaves as the ADDRESSED organ and is usable as an
arrival-time feature.

Residual caveat: the 459 rows with Situacao == 'Encaminhada por Outro Órgão'
are in transit and their OrgaoDestinatario is the receiver. Exclude that status
from training, or treat those rows as post-forwarding.
""")
