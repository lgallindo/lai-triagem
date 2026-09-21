"""Segunda rodada de verificação de vazamento, com as duas falhas metodológicas da
primeira rodada corrigidas.

H1  Disponibilidade de AssuntoPedido na chegada.
    Falha corrigida: rodar o teste de recência no ano CORRENTE (2026). No arquivo
    de 2024 toda linha já foi triada há ~2 anos, então "últimos 7 dias" nada
    dizia. No arquivo de 2026, linhas registradas dias antes do retrato de
    2026-09-14 são genuinamente recentes; se o SIC atribui o assunto durante a
    triagem, essas linhas têm de estar sem ele.

H2  OrgaoDestinatario guarda o órgão endereçado ou o final?
    Falha corrigida: abandonar o teste por prefixo de protocolo (pureza 0,58, o
    prefixo não identifica órgão) e usar três testes estruturais independentes:
      T1  ProtocoloPedido duplicado -> o reencaminhamento cria uma segunda
          linha, preservando a original, em vez de mutar uma linha no lugar?
      T2  Situacao x FoiReencaminhado -> a situação "Encaminhada por Outro
          Órgão" nomeia explicitamente o lado receptor.
      T3  Junção com Recursos -> Recursos carrega OrgaoPedido (o órgão do pedido
          tal como registrado no recurso) ao lado de Pedidos.OrgaoDestinatario.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lai_triagem.config import INTERIM  # noqa: E402
from lai_triagem.dados import limpar  # noqa: E402

SNAP = "20260914"
READ_KW = dict(sep=";", encoding="utf-16", dtype=str, na_values=[" ", ""], keep_default_na=True)
YEARS = [2022, 2023, 2024, 2025, 2026]


def _clean(df):
    df.columns = [c.strip() for c in df.columns]
    # P1: uma implementacao so, em lai_triagem/dados.py.
    return limpar(df)


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

    # Proxy de não triado: linhas ainda sem resposta.
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
    rec = rec[rec.Instancia.eq("Primeira Instância")]  # a 2ª instância escala para a CGU por desenho
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
