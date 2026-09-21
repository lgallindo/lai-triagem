"""
H6  A tabela `Solicitantes` traz o perfil NA ABERTURA do pedido, ou um retrato
    ATUAL replicado sobre pedidos antigos?

Auditoria externa observou 22.963 pessoas aparecendo em anos diferentes com
todos os campos demográficos idênticos, e não conseguiu resolver pela
documentação da CGU. Se for retrato atual, `Escolaridade` e `Profissao` são
POSTERIORES nas linhas antigas: alguém que se formou em 2024 apareceria formado
nos seus pedidos de 2022 -- vazamento clássico, a sexta família do projeto.

Previsões opostas e observáveis:

  perfil NA ABERTURA  -> ao longo de 5 anos, ALGUÉM tem de mudar de escolaridade
                         ou de profissão. Esperaríamos milhares de mudanças.
  retrato ATUAL       -> ZERO mudanças, porque é o mesmo registro copiado para
                         cada arquivo anual.

`DataNascimento` serve de controle: deve ser invariante nas duas hipóteses, e se
ela variar é sinal de que o identificador não é estável.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lai_triagem.config import INTERIM  # noqa: E402

KW = dict(sep=";", encoding="utf-16", dtype=str, na_values=[" ", ""], keep_default_na=True)
YEARS = [2022, 2023, 2024, 2025, 2026]
CAMPOS = ["TipoDemandante", "DataNascimento", "Genero", "Escolaridade",
          "Profissao", "TipoPessoaJuridica", "Pais", "UF", "Municipio"]


def load(year):
    hits = sorted(INTERIM.glob(f"*_SolicitantesPedidos_csv_{year}.csv"))
    if not hits:
        return None
    d = pd.read_csv(hits[-1], usecols=["IdSolicitante"] + CAMPOS, **KW)
    d.columns = [c.strip() for c in d.columns]
    for c in d.columns:
        d[c] = d[c].str.strip()
    d["ano"] = year
    return d.drop_duplicates("IdSolicitante")


frames = [f for f in (load(y) for y in YEARS) if f is not None]
df = pd.concat(frames, ignore_index=True)
df = df[df.IdSolicitante.ne("0")]

print(f"linhas de solicitante (dedup por ano): {len(df):,}")
print(f"solicitantes distintos: {df.IdSolicitante.nunique():,}")

# Quem aparece em mais de um ano
cnt = df.groupby("IdSolicitante").ano.nunique()
multi = cnt[cnt > 1].index
sub = df[df.IdSolicitante.isin(multi)]
print(f"solicitantes presentes em >1 ano: {len(multi):,}")
print(f"  linhas correspondentes: {len(sub):,}\n")

print("=" * 74)
print("TESTE PRINCIPAL — algum campo muda para a mesma pessoa entre anos?")
print("=" * 74)
print(f"  {'campo':<20} {'pessoas c/ >1 valor':>20} {'% dos multi-ano':>16}")
resultados = {}
for c in CAMPOS:
    nun = sub.groupby("IdSolicitante")[c].nunique(dropna=True)
    muda = int((nun > 1).sum())
    resultados[c] = muda
    print(f"  {c:<20} {muda:>20,} {100*muda/max(len(multi),1):>15.3f}%")

print(f"\n  controle `DataNascimento`: {resultados['DataNascimento']:,} mudanças")
print("    (deve ser ~0 nas duas hipóteses; se for alto, o id não é estável)")

mutaveis = ["Escolaridade", "Profissao"]
total_mut = sum(resultados[c] for c in mutaveis)
print(f"\n  campos que DEVERIAM mudar em 5 anos (escolaridade, profissão):"
      f" {total_mut:,} mudanças")

print("\n" + "=" * 74)
print("VEREDITO")
print("=" * 74)
if total_mut == 0:
    print("""  H6 CONFIRMADA — retrato ATUAL, não perfil na abertura.

  Zero mudanças de escolaridade ou profissão entre 2022 e 2026, para dezenas de
  milhares de pessoas, é impossível num perfil histórico: em cinco anos alguém
  se forma, muda de emprego. A ausência total de variação só se explica por um
  único registro por pessoa, replicado em cada arquivo anual.

  CONSEQUÊNCIA: `Escolaridade` e `Profissao` são POSTERIORES para as linhas
  antigas e não podem ser usadas como variáveis de chegada. Sexta família de
  vazamento do projeto.""")
elif total_mut < 0.001 * len(multi):
    print(f"""  H6 PROVAVELMENTE CONFIRMADA — variação desprezível ({total_mut:,} casos,
  {100*total_mut/len(multi):.4f}% dos multi-ano). Compatível com retrato atual
  mais um punhado de correções de cadastro.""")
else:
    print(f"""  H6 REFUTADA — há {total_mut:,} mudanças reais
  ({100*total_mut/len(multi):.2f}% dos multi-ano), compatível com perfil
  registrado na abertura de cada pedido.""")

# Teste complementar: as linhas são byte-idênticas entre arquivos?
print("\n" + "=" * 74)
print("TESTE COMPLEMENTAR — as linhas do 1º e do último ano são idênticas?")
print("=" * 74)
a, b = frames[0], frames[-1]
comum = set(a.IdSolicitante) & set(b.IdSolicitante)
print(f"  solicitantes em {YEARS[0]} e {YEARS[-1]}: {len(comum):,}")
if comum:
    ja = a[a.IdSolicitante.isin(comum)].set_index("IdSolicitante")[CAMPOS].sort_index()
    jb = b[b.IdSolicitante.isin(comum)].set_index("IdSolicitante")[CAMPOS].sort_index()
    iguais = int((ja.fillna("~") == jb.fillna("~")).all(axis=1).sum())
    print(f"  registros idênticos em todos os {len(CAMPOS)} campos: {iguais:,} "
          f"({100*iguais/len(comum):.3f}%)")
    if iguais == len(comum):
        print("  => 100% idênticos: o arquivo anual é o MESMO cadastro, recortado.")
