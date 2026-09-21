"""
H7 e H8 — dois defeitos levantados pela auditoria da camada 2 (18/09/2026).

O auditor externo esgotou a cota antes de medir; as hipóteses ficaram escritas
no log dele e são verificadas aqui, em vez de aceitas ou descartadas de boca.

**H7 — desalinhamento de índice no histórico do solicitante (suspeita GRAVE).**
`lagged_outcome_sums` devolve `m.index`, e `pd.merge_asof` **reinicia o índice**:
o que volta é `RangeIndex(0..n-1)`, ou seja **posição**, não rótulo de linha.
Quem consome faz `pd.Series(cum_y, index=order).reindex(sub.index)`, que procura
por **rótulo**. Como `df` levou `reset_index(drop=True)` (linha 254) e `sub`
exclui os solicitantes anônimos (`IdSolicitante == '0'`, 16,9% das linhas), os
rótulos de `sub` NÃO são iguais às posições: vão até `N-1`, enquanto as posições
param em `n-1`. Consequência esperada: parte das linhas recebe histórico de
**outra** linha, e parte recebe `NaN` (que depois vira `-1`, "sem histórico").

**H8 — prior não causal nas taxas móveis (suspeita MÉDIA).**
`base_all = float(df.y.mean())` (linha 331) é a taxa-base de **todo** o quadro,
2025 e 2026 inclusive, e entra como prior de suavização das janelas móveis
(linha 339). Para órgão com poucos pedidos na janela, o valor é dominado por
`PRIOR_MOVEL * base_all` — isto é, por uma quantidade calculada com rótulos de
validação e de teste.

    uv run python scripts/verify_h7_alinhamento.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train  # noqa: E402

from lai_triagem.variaveis_featuretools import (  # noqa: E402
    POR_PAR,
    POR_SOLICITANTE,
    desfechos_defasados,
)


def linha(t=""):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}" if t else "")


print("carregando coorte e construindo variáveis (usa o próprio train.py)...")
df = train.build_features(train.load_cohort(), train.organ_birth_table())
print(f"linhas: {len(df):,}")

# ---------------------------------------------------------------- H7
linha("H7 — o índice devolvido é posição ou rótulo?")

real = df.IdSolicitante.ne("0")
sub = df.loc[real]
print(f"linhas totais N          : {len(df):,}")
print(f"linhas com solicitante n : {len(sub):,}  ({100*len(sub)/len(df):.1f}%)")
print(f"df.index é RangeIndex?   : {isinstance(df.index, pd.RangeIndex)}")
print(f"sub.index é contíguo?    : {bool((np.diff(sub.index.to_numpy()) == 1).all())}")
print(f"maior rótulo em sub.index: {sub.index.max():,}")
print(f"maior posição possível   : {len(sub) - 1:,}")
if sub.index.max() > len(sub) - 1:
    print("  -> há rótulo além do fim das posições: `reindex` devolverá NaN "
          "para esses, e valor de OUTRA linha para os demais")

print("\nNESTE RAMO a pergunta acima mudou de natureza. `lagged_outcome_sums`")
print("não existe mais: quem calcula é `desfechos_defasados`, que pede ao")
print("Featuretools a agregação até `t - 60d`. O `merge_asof` sumiu, mas a")
print("JUNÇÃO NÃO: a biblioteca proíbe corte duplicado, então o resultado sai")
print("deduplicado e é recasado às linhas por (chave, corte). É a mesma classe")
print("de operação do H7, em outra roupa — e por isso continua sendo conferida")
print("aqui, contra a reconstrução sabidamente correta.")


# Reconstrução CORRETA: preserva o rótulo original antes do merge_asof.
def correto(sub, keys, lag_days=train.MATURITY_DAYS):
    daily = (sub.groupby(keys + ["_reg"], as_index=False, observed=True)
                .agg(dy=("y", "sum"), dn=("y", "size"))
                .sort_values("_reg", kind="stable"))
    daily["cum_y"] = daily.groupby(keys, observed=True).dy.cumsum()
    daily["cum_n"] = daily.groupby(keys, observed=True).dn.cumsum()
    left = sub[keys + ["_reg"]].copy()
    left["_cut"] = left["_reg"] - pd.Timedelta(days=lag_days)
    left = left.sort_values("_cut", kind="stable")
    rotulos = left.index.to_numpy()          # <- o que faltava guardar
    m = pd.merge_asof(left, daily[keys + ["_reg", "cum_y", "cum_n"]],
                      left_on="_cut", right_on="_reg", by=keys,
                      direction="backward", suffixes=("", "_d"))
    return (pd.Series(m.cum_y.fillna(0.0).to_numpy(), index=rotulos).reindex(sub.index),
            pd.Series(m.cum_n.fillna(0.0).to_numpy(), index=rotulos).reindex(sub.index))

for keys, alvo, rotulo in ((["IdSolicitante"], POR_SOLICITANTE, "prev_reenc_solicitante"),
                           (["IdSolicitante", "OrgaoDestinatario"], POR_PAR,
                            "prev_reenc_neste_orgao")):
    cy, cn = desfechos_defasados(df, sub, alvo)
    atual = pd.Series(cy, index=sub.index)                     # Featuretools
    certo, certo_n = correto(sub, keys)                        # referência à mão

    nan_atual = int(atual.isna().sum())
    nan_certo = int(certo.isna().sum())
    ambos = atual.notna() & certo.notna()
    difere = int((atual[ambos] != certo[ambos]).sum())
    soma_atual = float(atual.fillna(0).sum())
    soma_certo = float(certo.fillna(0).sum())

    linha(f"H7 — {rotulo}")
    print(f"NaN pelo Featuretools    : {nan_atual:,}")
    print(f"NaN pela referência      : {nan_certo:,}")
    print(f"ambos preenchidos, diferem: {difere:,}")
    print(f"soma dos desfechos, Featuretools: {soma_atual:,.0f}")
    print(f"soma dos desfechos, referência  : {soma_certo:,.0f}")
    if nan_atual > nan_certo or difere:
        print("  -> REGRESSÃO: o histórico está sendo atribuído às linhas erradas")
    else:
        print("  -> bate linha a linha. A junção de volta está correta HOJE; o")
        print("     que mudou é que ela é conferida, não que deixou de existir.")

# ---------------------------------------------------------------- H8
linha("H8 — o prior das taxas móveis usa rótulo de validação e teste?")

base_all = float(df.y.mean())
base_train = float(df.loc[df.ano.isin(train.TRAIN_YEARS), "y"].mean())
print(f"base_all   (todo o quadro, 2022-2026): {base_all:.6f}   <- usado hoje")
print(f"base_train (só anos de treino)       : {base_train:.6f}")
print(f"diferença                            : {base_all - base_train:+.6f}")

roll = train.organ_rolling(df)
for w in (90, 365):
    cnt, sm = roll[f"cnt_{w}"], roll[f"sum_{w}"]
    hoje = (sm + train.PRIOR_MOVEL * base_all) / (cnt + train.PRIOR_MOVEL)
    certo = (sm + train.PRIOR_MOVEL * base_train) / (cnt + train.PRIOR_MOVEL)
    d = np.abs(hoje - certo)
    print(f"\njanela {w}d: linhas afetadas {int((d > 0).sum()):,}"
          f"  |  desvio médio {float(d.mean()):.6f}"
          f"  |  desvio máximo {float(d.max()):.6f}")
    print(f"           acima de 0,001: {int((d > 0.001).sum()):,} linhas"
          f"  |  acima de 0,01: {int((d > 0.01).sum()):,}")

linha("veredito")
print("H7: ver as contagens acima — qualquer NaN a mais ou qualquer linha que")
print("    difere confirma desalinhamento, e o efeito é medido em seguida pelo")
print("    experimento de retreino.")
print("H8: o prior é calculado sobre todo o quadro, logo incorpora rótulo de")
print("    2025 e 2026. O desvio por linha diz se importa na prática.")
