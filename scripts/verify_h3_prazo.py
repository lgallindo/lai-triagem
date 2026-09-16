"""
H3  Is prazo_dias (PrazoAtendimento - DataRegistro) an arrival-time feature?

It is the trained model's dominant signal (40.5% of gain), so if Fala.BR
recalculates PrazoAtendimento when a request is prorrogated or forwarded, the
model is leaking and the headline numbers are void.

LAI sets 20 days, extendable once by 10 (art. 11 par. 2). Predictions:
  * NOT leaky  -> prazo_dias clusters at ~20 regardless of FoiProrrogado
                  and regardless of FoiReencaminhado.
  * leaky      -> prazo_dias is systematically larger when FoiProrrogado=Sim
                  (deadline rewritten after the extension was granted) and/or
                  differs by FoiReencaminhado.

H4  How usable are the demographic features at all?
    The equity audit showed Escolaridade 80.19% NaN. Quantify the join failure.
"""

from pathlib import Path

import numpy as np
import pandas as pd

INTERIM = Path.home() / "lai-triagem" / "data" / "interim"
SNAP = "20260914"
READ_KW = dict(sep=";", encoding="utf-16", dtype=str, na_values=[" ", ""], keep_default_na=True)
YEARS = [2022, 2023, 2024, 2025, 2026]


def _clean(df):
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    return df


frames = []
for y in YEARS:
    p = _clean(pd.read_csv(INTERIM / f"{SNAP}_Pedidos_csv_{y}.csv",
                           usecols=["IdPedido", "IdSolicitante", "DataRegistro", "PrazoAtendimento",
                                    "FoiProrrogado", "FoiReencaminhado", "Situacao"], **READ_KW))
    s = _clean(pd.read_csv(INTERIM / f"{SNAP}_SolicitantesPedidos_csv_{y}.csv",
                           usecols=["IdSolicitante", "Escolaridade", "Profissao", "Genero",
                                    "TipoDemandante"], **READ_KW)).drop_duplicates("IdSolicitante")
    d = p.merge(s, on="IdSolicitante", how="left")
    d["ano"] = y
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
reg = pd.to_datetime(df.DataRegistro, format="%d/%m/%Y", errors="coerce")
prazo = pd.to_datetime(df.PrazoAtendimento, format="%d/%m/%Y", errors="coerce")
df["prazo_dias"] = (prazo - reg).dt.days
df["reenc"] = df.FoiReencaminhado.eq("Sim")
df["prorr"] = df.FoiProrrogado.eq("Sim")

print("=" * 78)
print("H3  prazo_dias — arrival-time or rewritten after the fact?")
print("=" * 78)
print(f"rows {len(df):,}   prazo_dias non-null {df.prazo_dias.notna().sum():,}")
print(f"\noverall distribution of prazo_dias:")
print(df.prazo_dias.value_counts().head(12).sort_index().to_string())

print("\n-- by FoiProrrogado (the decisive split) --")
g = df.groupby("prorr").prazo_dias.agg(["count", "mean", "median",
                                        lambda s: s.quantile(0.05),
                                        lambda s: s.quantile(0.95)])
g.columns = ["n", "mean", "median", "p05", "p95"]
print(g.round(2).to_string())

print("\n-- by FoiReencaminhado --")
g2 = df.groupby("reenc").prazo_dias.agg(["count", "mean", "median"])
g2.columns = ["n", "mean", "median"]
print(g2.round(2).to_string())

print("\n-- cross-tab: median prazo_dias by (prorrogado x reencaminhado) --")
print(df.pivot_table(index="prorr", columns="reenc", values="prazo_dias",
                     aggfunc="median").round(1).to_string())

pn = df.loc[~df.prorr, "prazo_dias"].median()
py = df.loc[df.prorr, "prazo_dias"].median()
print(f"\nmedian prazo_dias: not-prorrogado {pn}  vs  prorrogado {py}   delta {py - pn:+.0f} days")
if pd.notna(py) and pd.notna(pn) and abs(py - pn) >= 5:
    print("=> VERDICT H3: LEAKY. PrazoAtendimento is rewritten when the extension is")
    print("   granted, which happens AFTER intake. prazo_dias must be dropped or")
    print("   replaced by a constant-at-arrival proxy.")
else:
    print("=> VERDICT H3: NOT leaky on the prorrogation axis; deadline is set at intake.")

# Does prazo_dias vary at all within a single organ+year? If it is purely
# statutory it should be near-constant, and its predictive power would then be
# coming from organ identity rather than from the deadline itself.
print("\n-- within-organ variability (is it just an organ proxy?) --")
sub = df[df.prazo_dias.notna()]
print(f"  distinct prazo_dias values: {sub.prazo_dias.nunique()}")
print(f"  share at the modal value:   {100*sub.prazo_dias.eq(sub.prazo_dias.mode()[0]).mean():.2f}%"
      f" (modal = {sub.prazo_dias.mode()[0]:.0f} days)")

print("\n" + "=" * 78)
print("H4  demographic feature availability (TAP scope depends on these)")
print("=" * 78)
print(f"IdSolicitante == '0' (anonymised): {df.IdSolicitante.eq('0').sum():,}"
      f"  ({100*df.IdSolicitante.eq('0').mean():.2f}%)")
for c in ("Escolaridade", "Profissao", "Genero", "TipoDemandante"):
    print(f"  {c:<16} missing {100*df[c].isna().mean():6.2f}%   distinct {df[c].nunique()}")

print("\n-- would the TAP's abstention rule fire? (abstain if profile missing) --")
ab = df.Escolaridade.isna() | df.Profissao.isna()
print(f"  rows with Escolaridade OR Profissao missing: {ab.sum():,} ({100*ab.mean():.2f}%)")
print("  => an abstain-on-missing-profile rule would decline to score that share of")
print("     all requests, which is not a viable triage product.")

print("\n-- is the missingness informative? reenc rate by profile availability --")
print(df.groupby(df.Escolaridade.isna()).reenc.agg(["count", "mean"]).rename(
    index={False: "profile present", True: "profile missing"}).round(4).to_string())
