"""
H2, final discriminating test.

T1/T3 established that there is exactly one row per ProtocoloPedido and that
Pedidos and Recursos agree perfectly. Neither settles ORIGINAL vs FINAL, because
both files come from the same 2026-09-14 snapshot and would mirror the same live
value if it were mutated.

This test discriminates on semantics instead of consistency.

The 459 rows with Situacao == "Encaminhada por Outro Órgão" are, by definition,
sitting at the RECEIVING organ right now (the status means "forwarded by another
organ"). So for those rows OrgaoDestinatario is provably the receiving organ.

If OrgaoDestinatario were rewritten to the receiver for ALL reencaminhados, then
the organs that rank high on reencaminhamento rate should look like the organs in
that in-transit set: receivers. If instead OrgaoDestinatario keeps the addressed
organ, high-reenc organs should be misaddressed generalist entry points, largely
DISJOINT from the receiver set.
"""

from pathlib import Path

import pandas as pd

INTERIM = Path.home() / "lai-triagem" / "data" / "interim"
SNAP = "20260914"
READ_KW = dict(sep=";", encoding="utf-16", dtype=str, na_values=[" ", ""], keep_default_na=True)
COLS = ["IdPedido", "OrgaoDestinatario", "FoiReencaminhado", "Situacao", "Esfera"]


def load_all():
    fr = []
    for y in (2022, 2023, 2024, 2025, 2026):
        d = pd.read_csv(INTERIM / f"{SNAP}_Pedidos_csv_{y}.csv", usecols=COLS, **READ_KW)
        d.columns = [c.strip() for c in d.columns]
        for c in d.columns:
            d[c] = d[c].str.strip()
        fr.append(d)
    return pd.concat(fr, ignore_index=True)


a = load_all()
a["reenc"] = a.FoiReencaminhado.eq("Sim")
print(f"rows {len(a):,}   base reenc rate {a.reenc.mean() * 100:.2f}%\n")

# Provable receivers: rows currently in transit.
receivers = set(a.loc[a.Situacao.eq("Encaminhada por Outro Órgão"), "OrgaoDestinatario"].dropna())
print(f"organs appearing as provable RECEIVERS (Situacao='Encaminhada por Outro Órgão'): {len(receivers)}")
print("  top 10 by in-transit volume:")
print(a[a.Situacao.eq("Encaminhada por Outro Órgão")].OrgaoDestinatario
      .value_counts().head(10).to_string(), "\n")

# Organs ranked by reencaminhamento rate (min volume so rates are meaningful).
g = a.groupby("OrgaoDestinatario").agg(n=("IdPedido", "size"), reenc=("reenc", "sum"))
g = g[g.n >= 500]
g["rate_%"] = (100.0 * g.reenc / g.n).round(2)
top = g.sort_values("rate_%", ascending=False).head(20)
print(f"top 20 organs by reencaminhamento rate (>=500 pedidos, {len(g)} organs qualify):")
top_ = top.copy()
top_["is_provable_receiver"] = [o in receivers for o in top_.index]
print(top_.to_string(), "\n")

overlap = top_.is_provable_receiver.sum()
print(f"overlap between high-reenc organs and provable receivers: {overlap}/20")
print("=> " + ("CONSISTENT WITH OVERWRITE: high-reenc organs are receivers"
              if overlap >= 12 else
              "CONSISTENT WITH 'ADDRESSED ORGAN': high-reenc organs are NOT the receivers"))

print("\nlowest 10 reenc-rate organs, for contrast:")
print(g.sort_values("rate_%").head(10).to_string())
