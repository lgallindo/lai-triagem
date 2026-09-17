"""H2, teste discriminante final.

T1/T3 estabeleceram que há exatamente uma linha por ProtocoloPedido e que
Pedidos e Recursos concordam perfeitamente. Nenhum dos dois decide entre ORIGINAL
e FINAL, porque ambos os arquivos vêm do mesmo retrato de 2026-09-14 e
espelhariam o mesmo valor corrente caso ele fosse mutado.

Este teste discrimina por semântica, não por consistência.

As 459 linhas com Situacao == "Encaminhada por Outro Órgão" estão, por
definição, paradas no órgão RECEPTOR neste momento (a situação significa
"encaminhada por outro órgão"). Logo, para essas linhas, OrgaoDestinatario é
comprovadamente o órgão receptor.

Se OrgaoDestinatario fosse reescrito para o receptor em TODOS os reencaminhados,
os órgãos com taxa alta de reencaminhamento deveriam parecer-se com os órgãos
desse conjunto em trânsito: receptores. Se, em vez disso, OrgaoDestinatario
mantém o órgão endereçado, os órgãos de taxa alta devem ser pontos de entrada
generalistas mal endereçados, em grande parte DISJUNTOS do conjunto de receptores.
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

# Receptores comprovados: linhas atualmente em trânsito.
receivers = set(a.loc[a.Situacao.eq("Encaminhada por Outro Órgão"), "OrgaoDestinatario"].dropna())
print(f"organs appearing as provable RECEIVERS (Situacao='Encaminhada por Outro Órgão'): {len(receivers)}")
print("  top 10 by in-transit volume:")
print(a[a.Situacao.eq("Encaminhada por Outro Órgão")].OrgaoDestinatario
      .value_counts().head(10).to_string(), "\n")

# Órgãos ordenados por taxa de reencaminhamento (volume mínimo para a taxa fazer sentido).
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
