"""
Construção de variáveis com **Featuretools**, em lugar da mecânica temporal à mão.

Este módulo substitui `organ_birth_table`, `organ_rolling`,
`lagged_outcome_sums` e `build_features`, que viviam em `scripts/train.py`. O
resto do treinamento — corte temporal, métrica, contrato de serviço, guardas —
não é tocado.

## O que se está testando

O Featuretools propõe `cutoff_time`: para cada linha-alvo declara-se um
instante, e toda agregação enxerga apenas o que existia até ali. Se a promessa
se confirmar, os pontos onde uma pessoa tem de acertar "não olhe para o futuro"
na mão caem para perto de zero — que é a medida que decide a comparação em
`docs/LINHA_BASE_COMPLEXIDADE.md`.

## O que ela de fato entrega, medido aqui

Confirma-se para as agregações **entre dias**. A janela de `training_window` é
`(corte - w, corte]` — fronteira inferior aberta, superior fechada, exatamente
a convenção que `organ_rolling` mantinha à mão com dois `searchsorted` e
`side="right"` escolhido a dedo. Sumiram também os `cumsum` e o `merge_asof`
dos desfechos defasados.

**Não** se confirma para quatro coisas, e as quatro moldaram este arquivo:

1. **O corte inclui o próprio instante, logo inclui o próprio rótulo.** Com
   `cutoff_time = t`, `SUM(eventos.y)` da linha em `t` soma o `y` dela mesma.
   Verificado na sonda: `COUNT` e `SUM` deram valores idênticos, porque cada
   evento entrava na própria agregação. Vazamento de alvo na configuração mais
   óbvia da biblioteca.
2. **Não dá para recuar o corte tendo o evento como alvo.** O reflexo seria
   `cutoff_time = t - ε`, mas aí o Featuretools considera que a linha-alvo
   ainda não existe e devolve zeros com o próprio `y` em `NaN` — para *todas*
   as linhas. O alvo teve de virar a tabela-PAI (solicitante, órgão, par), e
   não o pedido.
3. **Corte duplicado é erro, não aviso.** `(instância, tempo)` repetido levanta
   `AssertionError`. Como milhares de pedidos chegam ao mesmo órgão no mesmo
   dia, foi preciso **deduplicar e depois recasar o resultado com as linhas** —
   que é precisamente a junção cujo erro produziu o H7.
4. **O custo é por corte distinto, e a ordem intradiária é inviável.** Medido:
   ~6,3 ms por corte distinto. Um corte por linha, que é o que a ordem por
   `IdPedido` exigiria, daria mais de uma hora por passagem — e são sete. Por
   isso as contagens são calculadas em dois pedaços: o Featuretools responde
   "até a véspera" e o pandas responde "e mais estes, hoje, antes de mim".

O saldo honesto está em `docs/auditorias/2026-09-21-featuretools.md`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from lai_triagem.config import BIRTH_YEARS, COHORT, MATURITY_DAYS, PRIOR_MOVEL, TRAIN_YEARS
from lai_triagem.dados import arquivo_mais_recente, ler

# Avisos de terceiros silenciados, e vale dizer quantos: uma execução completa
# emitiu **7,6 MB** deles. São dois, os dois do `woodwork` reinicializando o
# esquema das tabelas a cada corte — um `pd.concat` com colunas vazias e um
# `Series.replace` — mais o `pkg_resources`, que o setuptools aposentou.
# Nenhum é do código deste projeto, e afogada nesse volume a saída do
# treinamento fica ilegível. Silenciados por módulo e por mensagem, nunca em
# bloco: um aviso vindo daqui tem de continuar aparecendo.
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="featuretools.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="woodwork.*")
import featuretools as ft  # noqa: E402

DIA = pd.Timedelta(days=1)

# Nomes das tabelas do `EntitySet`. Ficam em constantes porque aparecem tanto
# na montagem quanto no nome das colunas que o Featuretools gera
# (`COUNT(eventos)`, `SUM(eventos.y)`), e uma divergência entre os dois só
# apareceria como `KeyError` no fim de uma passagem de minutos.
EVENTOS = "eventos"
POR_SOLICITANTE = "sol_"
POR_ORGAO = "org_"
POR_PAR = "par_"


def _quadro_de_eventos(df: pd.DataFrame) -> pd.DataFrame:
    """A tabela de eventos que alimenta todas as agregações.

    Os instantes são **datas puras**, sem hora. É proposital: o corte do
    Featuretools é `<=`, então um corte na data `d` inclui todo evento
    registrado em `d`, que é exatamente o `side="right"` do código original.
    """
    return pd.DataFrame({
        "ev_id": np.arange(len(df), dtype="int64"),
        "t": df["_reg"].to_numpy(),
        "y": df["y"].to_numpy("float64"),
        # `dia` existe para o primitivo `Last` devolver um número de dia, e não
        # um carimbo: a diferença entre dois números de dia é exata, enquanto
        # `TIME_SINCE_LAST` mediria a partir do corte e não da data do pedido.
        "dia": df["_reg"].to_numpy("datetime64[D]").astype("int64"),
        "sol": df["IdSolicitante"].astype(str).to_numpy(),
        "org": df["OrgaoDestinatario"].astype(str).to_numpy(),
        "par": (df["IdSolicitante"].astype(str) + "\x1f"
                + df["OrgaoDestinatario"].astype(str)).to_numpy(),
    })


def monta_entityset(df: pd.DataFrame) -> ft.EntitySet:
    """Monta o `EntitySet`: eventos e as três tabelas-pai que os agrupam.

    Os pais existem porque o alvo de uma agregação **não pode ser o próprio
    evento** (ver item 2 do cabeçalho do módulo). Perguntar "quantos pedidos
    este solicitante já tinha em `d`" só funciona com o solicitante no papel de
    alvo.
    """
    eventos = _quadro_de_eventos(df)
    es = ft.EntitySet("lai")
    es = es.add_dataframe(
        dataframe_name=EVENTOS, dataframe=eventos, index="ev_id", time_index="t",
        logical_types={"sol": "Categorical", "org": "Categorical",
                       "par": "Categorical", "y": "Double", "dia": "Integer"})
    for nome, chave in ((POR_SOLICITANTE, "sol"), (POR_ORGAO, "org"), (POR_PAR, "par")):
        es = es.normalize_dataframe(base_dataframe_name=EVENTOS,
                                    new_dataframe_name=nome, index=chave)
    # Sem isto, `training_window` devolve **zero** em silêncio — não erro, não
    # aviso, zero. Verificado na sonda: a mesma consulta que dá 1, 2 e 3 com o
    # índice de último tempo dá 0, 0 e 0 sem ele. Numa biblioteca cuja proposta
    # é cuidar do tempo por você, a janela temporal é justamente o que falha
    # calado quando falta uma chamada de configuração.
    es.add_last_time_indexes()
    return es


def _agrega(es: ft.EntitySet, alvo: str, chaves: pd.Series, cortes: pd.Series,
            primitivas: list[str], colunas: list[str],
            janela: str | None = None) -> pd.DataFrame:
    """Roda uma passagem de DFS e devolve o resultado **na ordem das linhas**.

    Concentra aqui a deduplicação e o recasamento, que o Featuretools exige e
    que são o ponto frágil desta implementação. O corte repetido é proibido
    pela biblioteca, então manda-se cada par `(chave, corte)` uma vez só e
    depois junta-se de volta às linhas. Essa junção é a mesma operação que, em
    `merge_asof`, produziu o H7.

    A defesa possível está no `assert`: se a junção deixasse alguma linha sem
    resposta, a contagem não fecharia e a execução para. É verificação, não
    garantia — o H7 também "funcionava".
    """
    pedido = pd.DataFrame({"chave": chaves.to_numpy(), "corte": cortes.to_numpy()})
    unicos = pedido.drop_duplicates()
    entrada = pd.DataFrame({"instance_id": unicos["chave"].to_numpy(),
                            "time": unicos["corte"].to_numpy()})
    # `cutoff_time_in_index=True` é obrigatório aqui: sem ele o resultado volta
    # indexado só pela instância, e com a mesma instância repetida em cortes
    # diferentes não há como saber qual linha é qual — restaria confiar na
    # ordem, que é precisamente o tipo de suposição que produziu o H7.
    matriz, _ = ft.dfs(entityset=es, target_dataframe_name=alvo,
                       agg_primitives=primitivas, trans_primitives=[],
                       cutoff_time=entrada, training_window=janela,
                       cutoff_time_in_index=True, max_depth=1, verbose=False)
    matriz = matriz.reset_index()
    matriz = matriz.rename(columns={matriz.columns[0]: "chave", "time": "corte"})
    junto = pedido.merge(matriz[["chave", "corte", *colunas]], on=["chave", "corte"],
                         how="left", validate="many_to_one")
    assert len(junto) == len(pedido), (
        f"a junção de volta mudou o número de linhas: {len(junto)} contra "
        f"{len(pedido)}. Cortes duplicados ou chave ausente.")
    return junto[colunas]


def organ_birth_table() -> dict[str, pd.Timestamp]:
    """Primeira data de aparição de cada órgão, de 2012 em diante.

    Fica fora do Featuretools de propósito. Seria expressável — um `MIN` sobre
    a data, com corte no infinito —, mas montar um `EntitySet` de quinze anos
    de dados para tomar um mínimo custaria minutos e não removeria invariante
    nenhuma: não há janela nem rótulo envolvidos.
    """
    primeira: dict[str, pd.Timestamp] = {}
    for ano in BIRTH_YEARS + COHORT:
        if arquivo_mais_recente(ano) is None:
            continue
        d = ler(ano, ["OrgaoDestinatario", "DataRegistro"])
        d["_reg"] = pd.to_datetime(d.DataRegistro, format="%d/%m/%Y", errors="coerce")
        for orgao, dt in d.groupby("OrgaoDestinatario")._reg.min().items():
            if pd.notna(dt) and (orgao not in primeira or dt < primeira[orgao]):
                primeira[str(orgao)] = dt
    return primeira


def taxas_moveis_por_orgao(df: pd.DataFrame, janelas: tuple[int, ...] = (90, 365),
                           lag_dias: int = MATURITY_DAYS,
                           es: ft.EntitySet | None = None) -> dict[str, np.ndarray]:
    """Contagem e soma de `y` por órgão na janela móvel DEFASADA `(t-lag-w, t-lag]`.

    Substitui `organ_rolling`, e é onde o Featuretools se sai melhor. Os dois
    `searchsorted` com `side="right"`, o `cumsum` e a aritmética de índices
    `ycum[hi] - ycum[lo]` viram dois argumentos: o corte e a largura da janela.
    A fronteira `(corte - w, corte]` foi conferida na sonda e é a mesma que o
    original mantinha por convenção.

    A defasagem, essa continua manual. Nada na biblioteca sabe que o rótulo
    leva 60 dias para se firmar.
    """
    es = monta_entityset(df) if es is None else es
    # INVARIANTE: defasagem de maturação de 60 dias, escolhida à mão. O
    # `cutoff_time` aceita qualquer instante; que ele deva ficar 60 dias atrás
    # da data do pedido é decisão de quem modela, e nada verifica.
    corte = df["_reg"] - pd.Timedelta(days=lag_dias)
    saida: dict[str, np.ndarray] = {}
    for w in janelas:
        r = _agrega(es, POR_ORGAO, df["OrgaoDestinatario"].astype(str), corte,
                    ["count", "sum"], [f"COUNT({EVENTOS})", f"SUM({EVENTOS}.y)"],
                    janela=f"{w} days")
        saida[f"cnt_{w}"] = r[f"COUNT({EVENTOS})"].fillna(0).to_numpy("float64")
        saida[f"sum_{w}"] = r[f"SUM({EVENTOS}.y)"].fillna(0).to_numpy("float64")
    return saida


def desfechos_defasados(df: pd.DataFrame, sub: pd.DataFrame, chave: str,
                        lag_dias: int = MATURITY_DAYS,
                        es: ft.EntitySet | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Soma e contagem de `y` sobre o mesmo grupo até `t - lag_dias`.

    Substitui `lagged_outcome_sums`. Aquela função agregava por dia, tirava dois
    `cumsum`, fazia um `merge_asof` para trás e precisava **guardar o rótulo da
    linha antes do merge**, porque o `merge_asof` reinicia o índice — o H7, que
    deixou 94.145 pedidos sem histórico e deu a 120.060 o histórico de outra
    pessoa.

    O `cumsum` e o `merge_asof` sumiram. A junção de volta **não** sumiu: está
    em `_agrega`, exigida pela proibição de cortes duplicados.
    """
    es = monta_entityset(df) if es is None else es
    # INVARIANTE: a mesma defasagem, de novo à mão, num segundo lugar.
    corte = sub["_reg"] - pd.Timedelta(days=lag_dias)
    grupo = (sub["IdSolicitante"].astype(str) if chave == POR_SOLICITANTE
             else sub["IdSolicitante"].astype(str) + "\x1f"
             + sub["OrgaoDestinatario"].astype(str))
    r = _agrega(es, chave, grupo, corte, ["count", "sum"],
                [f"COUNT({EVENTOS})", f"SUM({EVENTOS}.y)"])
    return (r[f"SUM({EVENTOS}.y)"].fillna(0).to_numpy("float64"),
            r[f"COUNT({EVENTOS})"].fillna(0).to_numpy("float64"))


def contagens_do_solicitante(df: pd.DataFrame, sub: pd.DataFrame,
                             es: ft.EntitySet | None = None) -> dict[str, np.ndarray]:
    """Quantos pedidos o cidadão já fez, e há quanto tempo — sem tocar em rótulo.

    Estas variáveis não dependem de desfecho: que o cidadão já protocolou antes,
    inclusive hoje mais cedo, é fato conhecido na chegada. Por isso não levam
    defasagem — e por isso precisam da ordem **dentro do dia**, que é onde o
    Featuretools não chega.

    **A conta é feita em dois pedaços, e o motivo é medido.** O Featuretools
    responde "quantos até a véspera", com corte na data anterior; o pandas
    responde "e mais estes, hoje, antes de mim", com a ordem de `IdPedido`.
    Exprimir o segundo pedaço na biblioteca exigiria um corte distinto por
    linha, e o custo dela é por corte distinto: 6,3 ms medidos, vezes 655.177
    linhas, dá mais de uma hora — por passagem, e são sete.
    """
    es = monta_entityset(df) if es is None else es
    solicitante = sub["IdSolicitante"].astype(str)
    par = solicitante + "\x1f" + sub["OrgaoDestinatario"].astype(str)
    # INVARIANTE: o corte é a véspera, não o dia do pedido. Com o corte no
    # próprio dia o Featuretools inclui o pedido e todos os irmãos do mesmo dia
    # — o vazamento de mesmo dia, 159.320 linhas na auditoria externa.
    vespera = sub["_reg"] - DIA

    ate_ontem_sol = _agrega(es, POR_SOLICITANTE, solicitante, vespera,
                            ["count", "num_unique", "last"],
                            [f"COUNT({EVENTOS})", f"NUM_UNIQUE({EVENTOS}.org)",
                             f"LAST({EVENTOS}.dia)"])
    ate_ontem_par = _agrega(es, POR_PAR, par, vespera, ["count"], [f"COUNT({EVENTOS})"])

    n_sol = ate_ontem_sol[f"COUNT({EVENTOS})"].fillna(0).to_numpy("float64")
    n_par = ate_ontem_par[f"COUNT({EVENTOS})"].fillna(0).to_numpy("float64")
    distintos_ate_ontem = ate_ontem_sol[f"NUM_UNIQUE({EVENTOS}.org)"].fillna(0).to_numpy("float64")
    ultimo_dia = ate_ontem_sol[f"LAST({EVENTOS}.dia)"].to_numpy("float64")

    # INVARIANTE: a ordem dentro do dia, calculada no pandas porque o
    # Featuretools não a alcança a custo viável. `sub` já vem ordenado por
    # (`_reg`, `IdPedido`), e é dessa ordenação que estes `cumcount` dependem.
    dia = sub["_reg"]
    rank_sol = sub.groupby([solicitante, dia], sort=False).cumcount().to_numpy("float64")
    rank_par = sub.groupby([par, dia], sort=False).cumcount().to_numpy("float64")

    # INVARIANTE: é a soma dos dois pedaços que reconstrói a contagem. Se um
    # deles mudar de convenção — o corte virar o próprio dia, ou o `cumcount`
    # virar `cumcount() + 1` — o total continua plausível e fica errado.
    n_pedidos_previos = n_sol + rank_sol
    n_pedidos_previos_neste_orgao = n_par + rank_par

    # Primeira vez que este cidadão escreve a este órgão: nada até a véspera e
    # nenhum irmão mais cedo hoje.
    primeiro_par = ((n_par == 0) & (rank_par == 0)).astype("float64")
    # INVARIANTE: órgãos estreados HOJE, antes desta linha. O `NUM_UNIQUE` do
    # Featuretools para na véspera; sem esta parcela, dois pedidos do mesmo dia
    # para órgãos diferentes não se contariam.
    estreias_hoje = pd.Series(primeiro_par).groupby(
        [solicitante.to_numpy(), dia.to_numpy()], sort=False).cumsum().to_numpy() - primeiro_par
    n_orgaos_distintos_previos = distintos_ate_ontem + estreias_hoje

    # INVARIANTE: dias desde o último pedido. Quem tem irmão mais cedo hoje
    # responde zero; os demais, a diferença para o último dia com pedido. Sem o
    # ramo do mesmo dia, um reincidente do dia pareceria não ter histórico.
    dia_atual = sub["_reg"].to_numpy("datetime64[D]").astype("float64")
    dias_desde = np.where(rank_sol > 0, 0.0, dia_atual - ultimo_dia)
    dias_desde = np.where(n_pedidos_previos > 0, dias_desde, np.nan)

    return {
        "n_pedidos_previos": n_pedidos_previos,
        "n_pedidos_previos_neste_orgao": n_pedidos_previos_neste_orgao,
        "n_orgaos_distintos_previos": n_orgaos_distintos_previos,
        "dias_desde_ultimo_pedido": dias_desde,
    }


def build_features(df: pd.DataFrame, births: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Monta o quadro de variáveis. Mesmas colunas, mesmos valores, outra mecânica."""
    reg = pd.to_datetime(df.DataRegistro, format="%d/%m/%Y", errors="coerce")
    prazo = pd.to_datetime(df.PrazoAtendimento, format="%d/%m/%Y", errors="coerce")
    nasc = pd.to_datetime(df.DataNascimento, format="%d/%m/%Y", errors="coerce")
    df = df.assign(
        _reg=reg,
        prazo_dias=(prazo - reg).dt.days,   # só para a variante diagnóstica C
        reg_month=reg.dt.month, reg_dow=reg.dt.dayofweek, reg_day=reg.dt.day,
        idade=((reg - nasc).dt.days / 365.25).round(1),
        uf_match=(df.UF_sol.fillna("~") == df.UF.fillna("!")).astype("int8"),
        y=df.FoiReencaminhado.eq("Sim").astype("int8"),
    )
    # Idades implausíveis são erro de dado, não sinal.
    df.loc[(df.idade < 10) | (df.idade > 110), "idade"] = np.nan
    # Descarta linhas em trânsito: o OrgaoDestinatario delas é o receptor.
    df = df[df.Situacao.ne("Encaminhada por Outro Órgão")].copy()
    # INVARIANTE: ordem temporal real, com `IdPedido` desempatando dentro do
    # dia. É dela que dependem os `cumcount` intradiários de
    # `contagens_do_solicitante`; trocar a chave muda todas as contagens sem
    # que nada acuse. Verificado: correlação +0,996 com a data, 22 inversões
    # em 655.177.
    df["_idp"] = pd.to_numeric(df.IdPedido, errors="coerce")
    df = df.sort_values(["_reg", "_idp"], kind="stable").reset_index(drop=True)

    # Um `EntitySet` só, reaproveitado pelas sete passagens. Montá-lo custa
    # tempo e memória, e nada nele muda entre as consultas.
    es = monta_entityset(df)

    # --- histórico do solicitante --------------------------------------------
    # Solicitante anonimizado ('0') não acumula: não é uma pessoa. Ele fica no
    # `EntitySet` como um grupo à parte, que simplesmente nunca é consultado.
    real = df.IdSolicitante.ne("0")
    sub = df.loc[real]

    for col, v in contagens_do_solicitante(df, sub, es).items():
        df[col] = np.nan
        df.loc[real, col] = v.astype("float32")

    for col, chave in (("prev_reenc_solicitante", POR_SOLICITANTE),
                       ("prev_reenc_neste_orgao", POR_PAR)):
        cum_y, cum_n = desfechos_defasados(df, sub, chave, es=es)
        df[col] = np.nan
        df.loc[real, col] = cum_y.astype("float32")
        # A contagem madura acompanha, para a razão ter denominador coerente.
        df[col + "_den"] = np.nan
        df.loc[real, col + "_den"] = cum_n.astype("float32")

    # As razões usam o denominador MADURO, não a contagem total: dividir
    # desfechos maduros por contagem total subestimaria a taxa.
    df["prev_reenc_rate_solicitante"] = (
        df.prev_reenc_solicitante
        / df.prev_reenc_solicitante_den.where(df.prev_reenc_solicitante_den > 0)
    ).astype("float32")
    df["prev_reenc_rate_neste_orgao"] = (
        df.prev_reenc_neste_orgao
        / df.prev_reenc_neste_orgao_den.where(df.prev_reenc_neste_orgao_den > 0)
    ).astype("float32")
    # -1 marca "sem histórico", distinguível de zero, e é o que o chamador
    # deve enviar quando não tiver o dado.
    for c in ("n_pedidos_previos", "prev_reenc_solicitante",
              "prev_reenc_rate_solicitante", "n_pedidos_previos_neste_orgao",
              "prev_reenc_neste_orgao", "n_orgaos_distintos_previos",
              "dias_desde_ultimo_pedido", "prev_reenc_rate_neste_orgao"):
        df[c] = df[c].fillna(-1).astype("float32")

    # --- dinâmica do órgão ---------------------------------------------------
    roll = taxas_moveis_por_orgao(df, es=es)
    # H8: o prior tem de vir SÓ dos anos de treino. Com o prior de todo o
    # período, rótulo de validação e de teste entrava numa variável de entrada.
    base_prior = float(df.loc[df.ano.isin(TRAIN_YEARS), "y"].mean())
    for w in (90, 365):
        cnt, sm = roll[f"cnt_{w}"], roll[f"sum_{w}"]
        # Sem suavização, um órgão com um único pedido na janela produz taxa
        # 0,0 ou 1,0: 16,2% dos órgãos oscilavam mais de 5 pp por ruído.
        df[f"orgao_rate_movel_{w}d"] = (sm + PRIOR_MOVEL * base_prior) / (cnt + PRIOR_MOVEL)
    nascimento = df.OrgaoDestinatario.map(births)
    df["dias_desde_primeiro_pedido_do_orgao"] = (
        df._reg - nascimento).dt.days.astype("float32")
    return df
