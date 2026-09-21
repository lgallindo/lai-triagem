"""
H5  protocolo_seq (ProtocoloPedido[5:11]) é variável legítima de chegada?

Na ablação de engenharia de variáveis, essa única fatia de um identificador
elevou a PR-AUC de teste de 0,2018 para 0,4595 -- salto de 128%. Identificador
não deveria predizer nada. Pelo protocolo de docs/CAMPOS_POST_HOC.md, a variável
de maior ganho é a primeira suspeita, não a primeira comemoração.

Formato NUP presumido:  OOOOO SSSSSS AAAA DD  (órgão, sequencial, ano, dígitos)
Se a presunção estiver certa, ProtocoloPedido[11:15] tem de ser o ano de
DataRegistro. Se estiver errada, a fatia [5:11] contém outra coisa.

Testes:
  T1  o formato presumido se sustenta? [11:15] == ano do registro?
  T2  a taxa de reencaminhamento varia por decil de protocolo_seq?
  T3  ela varia DENTRO de um mesmo órgão e ano? (controla identidade do órgão)
  T4  o sequencial é realmente sequencial no tempo? (correlação com a data)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lai_triagem.config import INTERIM  # noqa: E402

READ_KW = dict(sep=";", encoding="utf-16", dtype=str, na_values=[" ", ""], keep_default_na=True)
COLS = ["ProtocoloPedido", "OrgaoDestinatario", "DataRegistro", "FoiReencaminhado", "Situacao"]


def load(years=(2022, 2023, 2024, 2025, 2026)):
    fr = []
    for y in years:
        hits = sorted(INTERIM.glob(f"*_Pedidos_csv_{y}.csv"))
        if not hits:
            continue
        d = pd.read_csv(hits[-1], usecols=COLS, **READ_KW)
        d.columns = [c.strip() for c in d.columns]
        for c in d.columns:
            d[c] = d[c].str.strip()
        d["ano"] = y
        fr.append(d)
    return pd.concat(fr, ignore_index=True)


df = load()
df = df[df.Situacao.ne("Encaminhada por Outro Órgão")].copy()
df["y"] = df.FoiReencaminhado.eq("Sim").astype("int8")
df["_reg"] = pd.to_datetime(df.DataRegistro, format="%d/%m/%Y", errors="coerce")
df = df[df.ProtocoloPedido.notna()].copy()

print("=" * 78)
print("T1  o formato NUP presumido se sustenta?")
print("=" * 78)
print("  distribuição de comprimento do protocolo:")
print(df.ProtocoloPedido.str.len().value_counts().to_string())
df["seg_ano"] = df.ProtocoloPedido.str.slice(11, 15)
ok = df.seg_ano.eq(df._reg.dt.year.astype(str))
print(f"\n  [11:15] == ano de DataRegistro em {100*ok.mean():.2f}% das linhas")
print("  => " + ("formato confirmado" if ok.mean() > 0.9 else
                 "FORMATO NÃO CONFIRMADO: a fatia [5:11] não é o sequencial que supus"))
print("\n  amostra de protocolos com a segmentação presumida:")
s = df.ProtocoloPedido.head(6)
for p in s:
    print(f"    {p}  ->  orgao={p[:5]} seq={p[5:11]} ano={p[11:15]} dv={p[15:17]}")

df["protocolo_seq"] = pd.to_numeric(df.ProtocoloPedido.str.slice(5, 11), errors="coerce")

print("\n" + "=" * 78)
print("T2  a taxa de reencaminhamento varia por decil de protocolo_seq?")
print("=" * 78)
df["dec"] = pd.qcut(df.protocolo_seq, 10, labels=False, duplicates="drop")
t2 = df.groupby("dec").agg(n=("y", "size"), reenc=("y", "mean"),
                           seq_min=("protocolo_seq", "min"),
                           seq_max=("protocolo_seq", "max"))
t2["reenc"] = t2.reenc.round(4)
print(t2.to_string())
spread = t2.reenc.max() - t2.reenc.min()
print(f"\n  amplitude entre decis: {spread:.4f} ({100*spread:.2f} pp)")

print("\n" + "=" * 78)
print("T3  varia DENTRO do mesmo órgão e ano? (controla identidade do órgão)")
print("=" * 78)
big = (df.groupby(["OrgaoDestinatario", "ano"]).size()
       .sort_values(ascending=False).head(6).index)
for organ, ano in big:
    sub = df[(df.OrgaoDestinatario == organ) & (df.ano == ano)].copy()
    if sub.protocolo_seq.nunique() < 10:
        continue
    sub["q"] = pd.qcut(sub.protocolo_seq, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    r = sub.groupby("q", observed=True).y.agg(["size", "mean"])
    r["mean"] = r["mean"].round(4)
    print(f"\n  {organ[:52]:<52} {ano}  n={len(sub):,}")
    print("    " + "  ".join(f"{q}={r.loc[q,'mean']:.4f}(n={r.loc[q,'size']})"
                             for q in r.index))

print("\n" + "=" * 78)
print("T4  o sequencial é sequencial no tempo?")
print("=" * 78)
for ano in (2024, 2025, 2026):
    sub = df[df.ano == ano].dropna(subset=["protocolo_seq", "_reg"])
    if len(sub) < 100:
        continue
    c = np.corrcoef(sub.protocolo_seq, sub._reg.astype("int64"))[0, 1]
    print(f"  {ano}: corr(protocolo_seq, data) = {c:+.4f}   n={len(sub):,}")

print("\n" + "=" * 78)
print("T5  quanto do sinal é só o prefixo (órgão) mal fatiado?")
print("=" * 78)
df["pref5"] = df.ProtocoloPedido.str.slice(0, 5)
print(f"  prefixos [0:5] distintos: {df.pref5.nunique():,}")
print(f"  órgãos distintos:         {df.OrgaoDestinatario.nunique():,}")
g = df.groupby("pref5").agg(n=("y", "size"), reenc=("y", "mean"),
                            orgaos=("OrgaoDestinatario", "nunique"))
g = g[g.n >= 500].sort_values("reenc", ascending=False)
print(f"\n  prefixos com >=500 pedidos: {len(g)}")
print("  10 prefixos de maior taxa:")
print(g.head(10).round(4).to_string())
print("\n  => se um prefixo mapeia muitos órgãos, [0:5] não é código de órgão;")
print("     mediana de órgãos por prefixo: "
      f"{g.orgaos.median():.0f}   máx: {g.orgaos.max():.0f}")
