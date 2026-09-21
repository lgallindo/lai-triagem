"""
Reajusta APENAS as tabelas por órgão do artefato, sem retreinar o modelo.

    uv run python scripts/refresh_organ_tables.py            # ensaio, não grava
    uv run python scripts/refresh_organ_tables.py --apply    # grava e re-registra

Motivação: `organ_rate_movel_90d` vale 16,13% do ganho do modelo e é, por
construção, um retrato. Sem reajuste ele envelhece e o valor se degrada
silenciosamente. Retreinar o modelo inteiro para atualizar uma tabela é
desperdício; e pior, muda o modelo, o que exige revalidação.

O QUE É REAJUSTADO
  organ_rate_movel_90d   taxa do órgão nos últimos 90 dias de dados disponíveis
  organ_rate_movel_365d  idem, 365 dias
  organ_birth            primeira aparição de cada órgão (capta órgãos novos)

O QUE NÃO É REAJUSTADO, E POR QUÊ
  orgao_rate      É codificação de alvo ajustada NOS ANOS DE TREINO. O modelo
                  aprendeu a mapear valores medidos naquela janela. Substituí-la
                  por uma medição recente desloca a distribuição da variável em
                  relação ao que foi treinado -- é distorção treino/serviço, a
                  mesma família de erro descrita em docs/CAMPOS_POST_HOC.md.
                  Ela só muda em retreinamento.
  category_codes  Mudar a codificação invalidaria as divisões das árvores.
  threshold       Calibrado como quantil dos escores de validação; depende do
                  modelo, não dos dados de órgão.
  o modelo        Intocado. Este script não treina nada.

DIFERENÇA DE JANELA ENTRE TREINO E REAJUSTE, que é sutil e importa

  No treinamento, a janela móvel tem de ser ESTRITAMENTE ANTERIOR à data de cada
  linha, senão a variável lê o próprio rótulo daquela linha.

  No reajuste, a janela usa TODO o histórico disponível até a data mais recente,
  inclusive. Não há vazamento: o pedido que será pontuado ainda não aconteceu,
  logo não está na janela. Tratar as duas situações igual seria erro nos dois
  sentidos -- vazamento no treino, ou informação jogada fora no serviço.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.home() / "lai-triagem"
INTERIM = ROOT / "data" / "interim"
ART = ROOT / "artifacts"
READ_KW = dict(sep=";", encoding="utf-16", dtype=str, na_values=[" ", ""], keep_default_na=True)
WINDOWS = (90, 365)
# Tem de ser idêntico ao PRIOR_MOVEL de scripts/train.py.
PRIOR_MOVEL = 20.0
# Só estes anos entram nas janelas móveis; os antigos servem para datar órgãos.
RECENT_YEARS = [2024, 2025, 2026]
BIRTH_YEARS = list(range(2012, 2027))

APPLY = "--apply" in sys.argv

# Chaves que este script tem permissão de tocar. Qualquer outra é erro.
REFRESHABLE = {"organ_rate_movel_90d", "organ_rate_movel_365d", "organ_birth",
               "organ_tables_refreshed_utc", "organ_tables_window_end"}


def _clean(df):
    df.columns = [c.strip() for c in df.columns]
    # H9: testar `dtype == object` NAO funciona -- READ_KW passa `dtype=str` e o
    # pandas moderno devolve o dtype `str` (PDEP-14). A condicao nunca era
    # verdadeira e a funcao era no-op. Ver scripts/train.py e
    # docs/auditorias/2026-09-18-camadas-2-3.md.
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip()
    return df


def _latest(year):
    hits = sorted(INTERIM.glob(f"*_Pedidos_csv_{year}.csv"))
    return hits[-1] if hits else None


def load(years, cols):
    fr = []
    for y in years:
        f = _latest(y)
        if f is None:
            continue
        d = _clean(pd.read_csv(f, usecols=cols, **READ_KW))
        d["_reg"] = pd.to_datetime(d.DataRegistro, format="%d/%m/%Y", errors="coerce")
        fr.append(d)
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def main():
    meta_path = ART / "preprocessor.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"artefato: {meta_path.name}   retrato de treino: {meta['data_snapshot']}")
    print(f"modo: {'APLICAR' if APPLY else 'ENSAIO (nada é gravado)'}\n")

    df = load(RECENT_YEARS, ["OrgaoDestinatario", "DataRegistro",
                             "FoiReencaminhado", "Situacao"])
    # Mesma exclusão do treinamento: linhas em trânsito têm o receptor no campo.
    df = df[df.Situacao.ne("Encaminhada por Outro Órgão")]
    df = df[df._reg.notna() & df.OrgaoDestinatario.notna()].copy()
    df["y"] = df.FoiReencaminhado.eq("Sim").astype("int8")

    end = df._reg.max()
    print(f"linhas recentes consideradas: {len(df):,}")
    print(f"fim da janela (data mais recente nos dados): {end.date()}")

    base = float(meta["base_rate"])
    new_tables = {}
    for w in WINDOWS:
        lo = end - pd.Timedelta(days=w)
        # Inclusiva no fim: o pedido a ser pontuado ainda não ocorreu, logo não
        # há como ele estar nesta janela. Ver o docstring do módulo.
        win = df[(df._reg > lo) & (df._reg <= end)]
        g = win.groupby("OrgaoDestinatario").y.agg(["sum", "count"])
        # Mesma suavização do treinamento (PRIOR_MOVEL em scripts/train.py).
        # Sem ela, órgão com um único pedido na janela dá taxa 0,0 ou 1,0.
        # Os dois valores TÊM de coincidir, senão o reajuste desloca a
        # distribuição da variável em relação ao que o modelo aprendeu.
        rate = ((g["sum"] + PRIOR_MOVEL * base) / (g["count"] + PRIOR_MOVEL)).astype(float)
        new_tables[f"organ_rate_movel_{w}d"] = rate.to_dict()
        print(f"\njanela {w}d ({lo.date()} a {end.date()}): "
              f"{len(win):,} pedidos, {len(rate):,} órgãos com volume")
        old = meta.get(f"organ_rate_movel_{w}d", {})
        common = set(old) & set(rate.index)
        if common:
            d = pd.Series({k: rate[k] - old[k] for k in common})
            print(f"  órgãos em comum com o artefato: {len(common):,}")
            print(f"  variação da taxa: média {d.mean():+.4f}  mediana {d.median():+.4f}  "
                  f"p05 {d.quantile(.05):+.4f}  p95 {d.quantile(.95):+.4f}")
            moved = (d.abs() > 0.05).sum()
            print(f"  órgãos que se moveram mais de 5 pp: {moved:,} ({100*moved/len(common):.1f}%)")
            worst = d.abs().sort_values(ascending=False).head(5)
            print("  maiores movimentos:")
            for organ in worst.index:
                print(f"    {organ[:54]:<54} {old[organ]:.4f} -> {rate[organ]:.4f}  ({d[organ]:+.4f})")
        print(f"  órgãos sem volume na janela recaem na taxa-base {base:.4f}: "
              f"{len(set(meta["organ_rate"]) - set(rate.index)):,}")

    # Nascimento dos órgãos: capta órgãos criados desde o último reajuste.
    births = {}
    bd = load(BIRTH_YEARS, ["OrgaoDestinatario", "DataRegistro"])
    for organ, dt in bd.groupby("OrgaoDestinatario")._reg.min().items():
        if pd.notna(dt):
            births[organ] = dt.strftime("%Y-%m-%d")
    old_b = meta.get("organ_birth", {})
    novos = sorted(set(births) - set(old_b))
    print(f"\norgan_birth: {len(births):,} órgãos datados "
          f"({len(novos)} novos desde o artefato)")
    for o in novos[:8]:
        print(f"  novo: {o[:58]:<58} nasceu {births[o]}")
    new_tables["organ_birth"] = births

    new_tables["organ_tables_refreshed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    new_tables["organ_tables_window_end"] = end.strftime("%Y-%m-%d")

    # Guarda-corpos: nada fora da lista permitida pode mudar.
    assert set(new_tables) <= REFRESHABLE, f"tentativa de alterar chave proibida: {set(new_tables) - REFRESHABLE}"
    print(f"\nchaves a gravar: {sorted(new_tables)}")
    print(f"chaves preservadas: organ_rate ({len(meta["organ_rate"])} entradas), "
          f"category_codes, threshold ({meta['threshold']}), feature_order, o modelo")

    if not APPLY:
        print("\nENSAIO — nada gravado. Reexecute com --apply.")
        return

    shutil.copy2(meta_path, meta_path.with_suffix(".json.bak"))
    meta.update(new_tables)
    # O artefato não pode ganhar dado pessoal por acidente.
    assert not any("solicit" in k.lower() for k in meta
                   if k != "caller_supplied_features"), "dado pessoal no artefato!"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"\ngravado {meta_path.name} ({meta_path.stat().st_size/1024:.0f} KB); "
          f"backup em {meta_path.with_suffix('.json.bak').name}")
    print("re-registre no BentoML para o serviço ver a tabela nova:")
    print("  uv run python scripts/register_bento.py")


if __name__ == "__main__":
    main()
