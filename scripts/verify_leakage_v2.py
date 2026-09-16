"""
Round 2 of leakage verification, with the two methodological gaps from round 1 closed.

H1  AssuntoPedido availability at arrival.
    Gap closed: run the recency test on the CURRENT year (2026). In the 2024 file
    every row has been triaged for ~2 years, so "last 7 days" said nothing. In the
    2026 file, rows registered days before the 2026-09-14 snapshot are genuinely
    fresh, so if the SIC assigns assunto during triage those rows must lack it.

H2  Does OrgaoDestinatario hold the originally addressed organ or the final one?
    Gap closed: drop the protocol-prefix test (purity 0.58, prefix is not an organ
    id) and use three independent structural tests instead:
      T1  duplicate ProtocoloPedido -> does forwarding create a second row,
          preserving the original, rather than mutating one row in place?
      T2  Situacao x FoiReencaminhado -> the status "Encaminhada por Outro Órgão"
          names the receiving side explicitly.
      T3  Recursos join -> Recursos carries OrgaoPedido (the pedido's organ as
          recorded on the appeal) next to Pedidos.OrgaoDestinatario.
"""

from pathlib import Path

import pandas as pd

INTERIM = Path.home() / "lai-triagem" / "data" / "interim"
SNAP = "20260914"
READ_KW = dict(sep=";", encoding="utf-16", dtype=str, na_values=[" ", ""], keep_default_na=True)
YEARS = [2022, 2023, 2024, 2025, 2026]


def _clean(df):
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def load(kind, year, cols=None):
    f = INTERIM / f"{SNAP}_{kind}_csv_{year}.csv"
    return _clean(pd.read_csv(f, usecols=cols, **READ_KW))


def pct(n, d):
    return f"{100.0 * n / d:.3f}%" if d else "n/a"


def rule(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# --------------------------------------------------------------------------- H1
def h1(year=2026):
    rule(f"H1  AssuntoPedido at arrival — CURRENT-YEAR test on {year}")
    df = load("Pedidos", year)
    n = len(df)
    df["_reg"] = pd.to_datetime(df.DataRegistro, format="%d/%m/%Y", errors="coerce")
    last = df._reg.max()
    print(f"rows {n:,}   latest DataRegistro {last.date()}   snapshot {SNAP}")
    print(f"gap between newest row and snapshot: {(pd.Timestamp('2026-09-14') - last).days} days\n")

    print("-- missingness of AssuntoPedido by Situacao --")
    g = df.groupby("Situacao", dropna=False).agg(
        rows=("IdPedido", "size"),
        miss=("AssuntoPedido", lambda s: s.isna().sum()),
    )
    g["miss_%"] = (100.0 * g.miss / g.rows).round(3)
    print(g.sort_values("rows", ascending=False).to_string(), "\n")

    print("-- recency gradient (now meaningful: these rows really are fresh) --")
    rows = []
    for days in (3, 7, 14, 30, 60, 90, 180, 365):
        t = df[df._reg > last - pd.Timedelta(days=days)]
        if len(t):
            rows.append((f"last {days}d", len(t), t.AssuntoPedido.isna().sum(),
                         100.0 * t.AssuntoPedido.isna().mean(),
                         100.0 * t.SubAssuntoPedido.isna().mean()))
    print(pd.DataFrame(rows, columns=["window", "rows", "assunto_miss", "assunto_miss_%",
                                      "subassunto_miss_%"]).to_string(index=False))

    # Untriaged proxy: rows with no response yet.
    open_rows = df[df.DataResposta.isna()]
    print(f"\nrows with no DataResposta (still open): {len(open_rows):,}"
          f"   assunto missing {pct(open_rows.AssuntoPedido.isna().sum(), len(open_rows))}")
    closed = df[df.DataResposta.notna()]
    print(f"rows answered:                          {len(closed):,}"
          f"   assunto missing {pct(closed.AssuntoPedido.isna().sum(), len(closed))}")


# --------------------------------------------------------------------------- H2
def h2_t1_duplicates():
    rule("H2 / T1  Does forwarding create a second row (original preserved)?")
    frames = []
    for y in YEARS:
        d = load("Pedidos", y, ["IdPedido", "ProtocoloPedido", "OrgaoDestinatario",
                                "FoiReencaminhado", "Situacao", "DataRegistro"])
        d["year"] = y
        frames.append(d)
    a = pd.concat(frames, ignore_index=True)
    print(f"total rows {len(a):,} across {YEARS}")
    print(f"distinct IdPedido       {a.IdPedido.nunique():,}")
    print(f"distinct ProtocoloPedido {a.ProtocoloPedido.nunique():,}")

    dup = a[a.duplicated("ProtocoloPedido", keep=False)].sort_values("ProtocoloPedido")
    print(f"\nrows sharing a ProtocoloPedido: {len(dup):,} "
          f"({dup.ProtocoloPedido.nunique():,} distinct protocols)")
    if len(dup):
        per = dup.groupby("ProtocoloPedido").agg(
            n=("IdPedido", "size"),
            n_organs=("OrgaoDestinatario", "nunique"),
            reenc=("FoiReencaminhado", lambda s: (s == "Sim").sum()),
        )
        print(f"  protocols whose duplicate rows sit at >1 organ: {(per.n_organs > 1).sum():,} "
              f"({pct((per.n_organs > 1).sum(), len(per))})")
        print(f"  of those, at least one row flagged reencaminhado: "
              f"{((per.n_organs > 1) & (per.reenc > 0)).sum():,}")
        ex = per[per.n_organs > 1].head(3).index
        for p in ex:
            print(f"\n  example protocol {p}:")
            print(dup[dup.ProtocoloPedido == p][
                ["year", "IdPedido", "OrgaoDestinatario", "FoiReencaminhado", "Situacao"]
            ].to_string(index=False))
    return a


def h2_t2_situacao(a):
    rule("H2 / T2  Situacao x FoiReencaminhado")
    ct = pd.crosstab(a.Situacao, a.FoiReencaminhado, dropna=False)
    ct["reenc_%"] = (100.0 * ct.get("Sim", 0) / ct.sum(axis=1)).round(2)
    print(ct.sort_values("reenc_%", ascending=False).to_string())


def h2_t3_recursos(year=2024):
    rule(f"H2 / T3  Recursos join — Pedidos.OrgaoDestinatario vs Recursos.OrgaoPedido ({year})")
    ped = load("Pedidos", year, ["IdPedido", "OrgaoDestinatario", "FoiReencaminhado", "Situacao"])
    rec = load("Recursos_Reclamacoes", year,
               ["IdPedido", "OrgaoPedido", "OrgaoDestinatario", "Instancia"])
    rec = rec[rec.Instancia.eq("Primeira Instância")]  # 2nd instance escalates to CGU by design
    print(f"pedidos {len(ped):,}   recursos 1a instancia {len(rec):,}")

    m = rec.merge(ped, on="IdPedido", suffixes=("_rec", "_ped"), how="inner")
    print(f"joined on IdPedido: {len(m):,}")
    if not len(m):
        print("  no overlap (appeals reference pedidos from earlier years) — test void")
        return

    m["match"] = m.OrgaoPedido.eq(m.OrgaoDestinatario_ped)
    for label, sub in (("NOT reencaminhado", m[m.FoiReencaminhado.ne("Sim")]),
                       ("reencaminhado", m[m.FoiReencaminhado.eq("Sim")])):
        if len(sub):
            print(f"  {label:<20} n={len(sub):>7,}   "
                  f"Pedidos.OrgaoDestinatario == Recursos.OrgaoPedido: {pct(sub.match.sum(), len(sub))}")

    r = m[m.FoiReencaminhado.eq("Sim")]
    c = m[m.FoiReencaminhado.ne("Sim")]
    if len(r) and len(c):
        drop = 100 * (c.match.mean() - r.match.mean())
        print(f"\n  agreement drop on reencaminhado: {drop:.2f} pp")
        print("  => " + ("OrgaoDestinatario differs from the appeal's OrgaoPedido on "
                         "reencaminhados; consistent with ONE of them being rewritten"
                         if abs(drop) > 5 else
                         "no differential; both fields agree regardless of reencaminhamento"))
        mism = r[~r.match]
        if len(mism):
            print("\n  sample mismatches (reencaminhado):")
            print(mism[["IdPedido", "OrgaoPedido", "OrgaoDestinatario_ped"]].head(5).to_string(index=False))


if __name__ == "__main__":
    h1(2026)
    allp = h2_t1_duplicates()
    h2_t2_situacao(allp)
    h2_t3_recursos(2024)
    h2_t3_recursos(2025)
