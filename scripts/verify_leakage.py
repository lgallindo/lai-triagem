"""
Verification of two leakage hypotheses on the CGU Fala.BR LAI "Pedidos" dataset.

H1  AssuntoPedido / SubAssuntoPedido are assigned by the SIC during triage,
    i.e. they are a triage OUTPUT and not available at request-arrival time.

H2  OrgaoDestinatario is overwritten when a request is reencaminhado,
    i.e. it holds the FINAL recipient rather than the originally addressed organ.

H2 test relies on the Brazilian NUP protocol layout:
    ProtocoloPedido = OOOOO SSSSSS YYYY DD   (organ, sequence, year, check digits)
The 5-digit organ prefix is fixed at registration. If OrgaoDestinatario were
rewritten on reencaminhamento while the protocol kept the original organ, the
prefix -> organ mapping must degrade specifically on reencaminhado rows.
"""

import sys
from pathlib import Path

import pandas as pd

INTERIM = Path.home() / "lai-triagem" / "data" / "interim"
SNAP = "20260914"
READ_KW = dict(sep=";", encoding="utf-16", dtype=str, na_values=[" ", ""], keep_default_na=True)

ARRIVAL_DOC = [
    "Esfera", "UF", "Municipio", "OrgaoDestinatario", "DataRegistro",
    "PrazoAtendimento", "FormaResposta", "OrigemSolicitacao",
]
POST_HOC = [
    "Situacao", "FoiProrrogado", "DataResposta", "Decisao", "EspecificacaoDecisao",
    "DetalhamentoDecisao", "MotivoNegativaAcesso", "PrazoRestricaoAcesso",
]


def load_pedidos(year: int) -> pd.DataFrame:
    df = pd.read_csv(INTERIM / f"{SNAP}_Pedidos_csv_{year}.csv", **READ_KW)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def pct(n, d):
    return f"{100.0 * n / d:.3f}%" if d else "n/a"


def h1_assunto(df: pd.DataFrame, year: int) -> None:
    print(f"\n{'=' * 78}\nH1  AssuntoPedido availability at arrival — {year}\n{'=' * 78}")
    n = len(df)
    print(f"rows: {n:,}")

    for col in ["AssuntoPedido", "SubAssuntoPedido", "Tag"]:
        miss = df[col].isna().sum()
        print(f"  {col:<20} missing {miss:>9,}  ({pct(miss, n)})   distinct {df[col].nunique():>6,}")

    # Decisive slice: requests still open. If the SIC assigns assunto during
    # triage, open requests that have not been triaged yet must lack it.
    print("\n  -- missingness of AssuntoPedido by Situacao --")
    g = df.groupby("Situacao", dropna=False).agg(
        rows=("IdPedido", "size"),
        assunto_missing=("AssuntoPedido", lambda s: s.isna().sum()),
    )
    g["missing_%"] = (100.0 * g.assunto_missing / g.rows).round(3)
    print(g.sort_values("rows", ascending=False).to_string())

    # Recency: newest registrations are the least likely to have been triaged.
    d = pd.to_datetime(df["DataRegistro"], format="%d/%m/%Y", errors="coerce")
    df = df.assign(_reg=d)
    last = d.max()
    print(f"\n  latest DataRegistro in file: {last.date() if pd.notna(last) else 'n/a'}")
    for days in (7, 30, 90):
        tail = df[df._reg > last - pd.Timedelta(days=days)]
        if len(tail):
            miss = tail["AssuntoPedido"].isna().sum()
            print(f"  last {days:>3}d: rows {len(tail):>7,}  assunto missing {miss:>6,} ({pct(miss, len(tail))})")

    print("\n  -- post-hoc columns, for contrast (known to be filled only after response) --")
    for col in POST_HOC:
        if col in df.columns:
            miss = df[col].isna().sum()
            print(f"  {col:<24} missing {pct(miss, n)}")


def h2_orgao(df: pd.DataFrame, year: int) -> None:
    print(f"\n{'=' * 78}\nH2  OrgaoDestinatario: original vs final recipient — {year}\n{'=' * 78}")

    d = df[df.ProtocoloPedido.notna() & df.OrgaoDestinatario.notna()].copy()
    d["prefix"] = d.ProtocoloPedido.str[:5]
    d["reenc"] = d.FoiReencaminhado.eq("Sim")

    print(f"rows usable: {len(d):,}   reencaminhado: {d.reenc.sum():,} ({pct(d.reenc.sum(), len(d))})")
    print(f"distinct protocol prefixes: {d.prefix.nunique():,}   distinct organs: {d.OrgaoDestinatario.nunique():,}")

    # Is the prefix actually an organ identifier? Measure how concentrated the
    # organ distribution is within each prefix, on NON-reencaminhado rows only
    # (those were never forwarded, so prefix and organ must agree if the
    # prefix encodes the addressed organ).
    clean = d[~d.reenc]
    if not len(clean):
        print("  no non-reencaminhado rows; cannot calibrate")
        return

    modal = clean.groupby("prefix").OrgaoDestinatario.agg(lambda s: s.value_counts().idxmax())
    cov = clean.groupby("prefix").OrgaoDestinatario.agg(lambda s: s.value_counts().iloc[0] / len(s))
    weight = clean.groupby("prefix").size()
    weighted_purity = (cov * weight).sum() / weight.sum()
    print(f"\n  prefix->organ purity on non-reencaminhado rows (weighted): {weighted_purity:.4f}")
    print(f"  => prefix {'IS' if weighted_purity > 0.9 else 'is NOT'} a reliable organ identifier")

    if weighted_purity <= 0.9:
        print("  H2 test inconclusive via protocol prefix; falling back to Recursos join only.")
        return

    # Now the actual test: does the prefix's modal organ still match
    # OrgaoDestinatario on reencaminhado rows?
    d["expected"] = d.prefix.map(modal)
    known = d[d.expected.notna()]
    agree = known.expected.eq(known.OrgaoDestinatario)

    for label, sub in (("NOT reencaminhado", known[~known.reenc]), ("reencaminhado", known[known.reenc])):
        a = agree[sub.index]
        print(f"  {label:<20} n={len(sub):>8,}   protocol-organ agreement: {pct(a.sum(), len(sub))}")

    a_clean = agree[known[~known.reenc].index].mean()
    a_reenc = agree[known[known.reenc].index].mean() if known.reenc.any() else float("nan")
    print(f"\n  agreement drop on reencaminhado rows: {100 * (a_clean - a_reenc):.2f} pp")
    if pd.notna(a_reenc) and (a_clean - a_reenc) > 0.10:
        print("  => VERDICT: OrgaoDestinatario appears OVERWRITTEN on reencaminhamento (post-hoc, leaky)")
    else:
        print("  => VERDICT: no evidence of overwrite; OrgaoDestinatario looks like the ADDRESSED organ")


def main() -> None:
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    df = load_pedidos(year)
    print(f"columns ({len(df.columns)}): {list(df.columns)}")
    h1_assunto(df, year)
    h2_orgao(df, year)


if __name__ == "__main__":
    main()
