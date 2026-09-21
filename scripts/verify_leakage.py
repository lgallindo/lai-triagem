"""Verificação de duas hipóteses de vazamento no conjunto "Pedidos" da LAI (Fala.BR/CGU).

H1  AssuntoPedido / SubAssuntoPedido são atribuídos pelo SIC durante a triagem,
    isto é, são SAÍDA da triagem e não estão disponíveis quando o pedido chega.

H2  OrgaoDestinatario é sobrescrito quando o pedido é reencaminhado, ou seja,
    guarda o destinatário FINAL em vez do órgão originalmente endereçado.

O teste de H2 apoia-se no formato do protocolo NUP brasileiro:
    ProtocoloPedido = OOOOO SSSSSS AAAA DD   (órgão, sequencial, ano, dígitos)
O prefixo de 5 dígitos do órgão é fixado no registro. Se OrgaoDestinatario
fosse reescrito no reencaminhamento enquanto o protocolo mantivesse o órgão
original, o mapeamento prefixo -> órgão teria de degradar especificamente nas
linhas reencaminhadas.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lai_triagem.config import INTERIM  # noqa: E402

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

    # Corte decisivo: pedidos ainda abertos. Se o SIC atribui o assunto durante
    # a triagem, pedidos abertos e não triados têm de estar sem ele.
    print("\n  -- missingness of AssuntoPedido by Situacao --")
    g = df.groupby("Situacao", dropna=False).agg(
        rows=("IdPedido", "size"),
        assunto_missing=("AssuntoPedido", lambda s: s.isna().sum()),
    )
    g["missing_%"] = (100.0 * g.assunto_missing / g.rows).round(3)
    print(g.sort_values("rows", ascending=False).to_string())

    # Recência: os registros mais novos são os menos prováveis de já triados.
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

    # O prefixo é de fato identificador de órgão? Mede a concentração da
    # distribuição de órgãos dentro de cada prefixo, apenas nas linhas NÃO
    # reencaminhadas (nunca foram encaminhadas, então prefixo e órgão têm de
    # concordar se o prefixo codificar o órgão endereçado).
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

    # Agora o teste de fato: o órgão modal do prefixo ainda coincide com
    # OrgaoDestinatario nas linhas reencaminhadas?
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
