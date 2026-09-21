"""
Construção de variáveis com **Temporian**, em lugar da mecânica temporal à mão.

Este módulo substitui `organ_birth_table`, `organ_rolling`,
`lagged_outcome_sums` e `build_features`, que viviam em `scripts/train.py`. O
resto do treinamento — corte temporal, métrica, contrato de serviço, guardas —
não é tocado.

## O que se está testando

A promessa do Temporian é que uma variável **não consegue** depender do futuro:
toda operação móvel é definida sobre a janela `(t - w, t]`, e olhar adiante
exige pedir explicitamente com `leak()`. Se a promessa se confirmar, os pontos
onde uma pessoa tem de acertar "não olhe para o futuro" na mão caem para perto
de zero — que é a medida que decide a comparação em
`docs/LINHA_BASE_COMPLEXIDADE.md`.

## O que ela de fato entrega, medido aqui

Confirma-se para a **fronteira superior da janela**: `moving_sum` e
`moving_count` jamais alcançam um evento posterior ao instante amostrado. Os
dois `searchsorted` com `side="right"` de `organ_rolling`, o `merge_asof` de
`lagged_outcome_sums` e os `cumsum` que os acompanhavam desaparecem, e com eles
a classe do H7 — não há índice a recasar, porque o resultado volta preso à
amostragem que o gerou.

**Não** se confirma para três coisas, e vale dizer quais:

1. **O evento vê a si mesmo.** Em `(t - w, t]` o `t` está dentro. Medido na
   sonda: dois eventos no mesmo instante somam um ao outro *e a si próprios*.
   Como `DataRegistro` é data sem hora, todo pedido do mesmo dia é simultâneo —
   é exatamente o vazamento de mesmo dia que a auditoria externa mediu em
   159.320 linhas. O Temporian não o impede; ele o reproduz fielmente, porque
   para ele um evento em `t` já aconteceu em `t`.
2. **A defasagem de maturação é escolha de quem modela.** Nada na biblioteca
   sabe que `FoiReencaminhado` leva 60 dias para se firmar. A defasagem
   continua entrando à mão, no instante de amostragem.
3. **O desempate intradiário.** O Temporian conhece um instante por evento, e o
   projeto precisa de duas convenções ao mesmo tempo: as janelas por órgão
   comparam por *data* (todo pedido daquele dia entra), e as contagens do
   solicitante comparam por *ordem de registro* (`IdPedido` desempata dentro do
   dia). Isso obriga a construir dois conjuntos de eventos com relógios
   diferentes — ver `PASSO_DESEMPATE`.

O saldo honesto está em `docs/auditorias/2026-09-21-temporian.md`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import temporian as tp

from lai_triagem.config import BIRTH_YEARS, COHORT, MATURITY_DAYS, PRIOR_MOVEL, TRAIN_YEARS
from lai_triagem.dados import arquivo_mais_recente, ler

SEGUNDOS_POR_DIA = 86_400.0

# Janela que cobre todo o período com folga, usada onde o original acumulava
# desde o começo. O Temporian não tem "janela infinita"; 200 anos é maior que
# qualquer distância entre dois pedidos da coorte, e nomear a constante evita
# um `1e9` solto que ninguém sabe de onde veio.
JANELA_TOTAL = 200 * 365 * SEGUNDOS_POR_DIA

# Desempate intradiário, em segundos.
#
# `DataRegistro` é data sem hora, então todos os pedidos de um dia caem no mesmo
# instante — e para o Temporian eventos simultâneos enxergam uns aos outros. As
# contagens do solicitante precisam da ordem real de chegada, que é `IdPedido`
# (correlação +0,996 com a data, 22 inversões em 655.177). Somamos à data um
# milésimo de segundo por posição na ordem global já ordenada por
# (`_reg`, `IdPedido`), o que torna os instantes estritamente crescentes sem
# jamais cruzar a meia-noite: são 655.177 linhas, logo no máximo 656 segundos
# de deslocamento contra os 86.400 de um dia.
#
# Um milésimo de segundo é folgado para o `float64`: na escala de 1,8e9 segundos
# (datas de 2022 a 2026) o menor incremento representável é ~2,4e-7 s, quatro
# mil vezes menor que o passo.
PASSO_DESEMPATE = 1e-3


def _segundos(reg: pd.Series) -> np.ndarray:
    """Converte carimbos do pandas em segundos desde a época, como `float64`."""
    return reg.to_numpy("datetime64[s]").astype("int64").astype("float64")


def _texto(coluna: pd.Series) -> np.ndarray:
    """Converte uma coluna de texto no vetor de `str` que o Temporian indexa."""
    return np.asarray(coluna.to_numpy(), dtype=np.str_)


def _para_pandas(ev: tp.EventSet) -> pd.DataFrame:
    """Converte para pandas DEPOIS de descartar o índice.

    `tp.to_pandas` faz `astype(str)` sobre a chave de índice, que o Temporian
    guarda como `bytes`, e o faz sem informar a codificação — ou seja, em ASCII.
    Todo nome de órgão do Fala.BR tem a forma `SIGLA – Nome`, com travessão
    (U+2013), e o travessão derruba a conversão com `UnicodeDecodeError`. Não é
    caso de borda: são 877 órgãos, praticamente todos.

    Descartar o índice antes de converter resolve porque nada aqui precisa dele
    de volta: a linha de origem viaja no `rid`, não na chave.
    """
    return tp.to_pandas(ev.drop_index(keep=False))


def _na_ordem_das_linhas(tabela: pd.DataFrame, colunas: list[str],
                         n: int) -> dict[str, np.ndarray]:
    """Devolve as colunas de `tabela` reordenadas pela linha de origem.

    O Temporian devolve os eventos agrupados por índice, não na ordem em que
    entraram. Em vez de tentar recompor essa ordem por (índice, instante) — que
    é ambíguo quando dois pedidos do mesmo órgão caem no mesmo dia —, cada
    consulta leva junto um `rid`, a posição da linha no quadro. É o mesmo
    problema que produziu o H7, resolvido de um jeito que não tem como errar em
    silêncio: se algum `rid` sumisse, o `assert` abaixo acusaria.
    """
    rid = tabela["rid"].to_numpy().astype("int64")
    assert len(rid) == n and len(np.unique(rid)) == n, (
        f"a consulta ao Temporian devolveu {len(rid)} eventos para {n} linhas; "
        f"não há correspondência de um para um")
    saida = {}
    for c in colunas:
        v = np.zeros(n, dtype="float64")
        v[rid] = tabela[c].to_numpy().astype("float64")
        saida[c] = v
    return saida


def organ_birth_table() -> dict[str, pd.Timestamp]:
    """Primeira data de aparição de cada órgão, de 2012 em diante.

    Fica fora do Temporian de propósito: é um mínimo sobre datas, sem janela e
    sem rótulo, logo não há fronteira temporal a garantir. Passar por um
    `EventSet` só para tomar um mínimo acrescentaria conversão sem remover
    invariante nenhuma.
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
                           lag_dias: int = MATURITY_DAYS) -> dict[str, np.ndarray]:
    """Contagem e soma de `y` por órgão na janela móvel DEFASADA `(t-lag-w, t-lag]`.

    Substitui `organ_rolling`. Onde havia dois `searchsorted` com `side="right"`
    escolhido à mão, um `cumsum` e a aritmética de índices `ycum[hi] - ycum[lo]`,
    há agora uma amostragem: pergunta-se ao Temporian o que ele sabia em
    `t - lag`, e a janela `(., t-lag]` é definição da operação, não convenção
    que alguém precisa lembrar.

    A defasagem, essa continua manual — é o `- lag_dias` abaixo. O Temporian não
    tem como saber que o rótulo leva 60 dias para se firmar.

    Os instantes aqui são **datas puras**, sem o desempate intradiário: o
    original compara por data, de modo que todo pedido registrado no dia
    `t - lag` entra na janela. Acrescentar o desempate mudaria a fronteira.
    """
    n = len(df)
    ts = _segundos(df["_reg"])
    orgao = _texto(df["OrgaoDestinatario"])
    rid = np.arange(n, dtype="float64")

    fonte = tp.event_set(timestamps=ts,
                         features={"y": df["y"].to_numpy("float64"), "orgao": orgao},
                         indexes=["orgao"])
    # INVARIANTE: defasagem de maturação de 60 dias, escolhida à mão. Nada no
    # Temporian sabe que o rótulo leva esse tempo para se firmar; se este
    # `- lag_dias` sumir, a variável passa a ler desfecho que em produção ainda
    # não existiria, e nenhuma operação da biblioteca reclama.
    consulta = tp.event_set(timestamps=ts - lag_dias * SEGUNDOS_POR_DIA,
                            features={"rid": rid, "orgao": orgao},
                            indexes=["orgao"])

    saida: dict[str, np.ndarray] = {}
    for w in janelas:
        # INVARIANTE: estes instantes são datas puras, SEM o desempate
        # intradiário usado nas contagens. É essa ausência que faz todo pedido
        # do dia `t - lag` entrar na janela, reproduzindo o `side="right"` do
        # original. Acrescentar o desempate aqui mudaria a fronteira em
        # silêncio, e o resultado continuaria plausível.
        largura = w * SEGUNDOS_POR_DIA
        soma = fonte["y"].moving_sum(window_length=largura, sampling=consulta)
        cont = fonte["y"].moving_count(window_length=largura, sampling=consulta)
        tabela = _para_pandas(tp.glue(soma.rename(f"sum_{w}"),
                                      cont.rename(f"cnt_{w}").cast(tp.float64),
                                      consulta["rid"]))
        saida.update(_na_ordem_das_linhas(tabela, [f"sum_{w}", f"cnt_{w}"], n))
    return saida


def desfechos_defasados(sub: pd.DataFrame, chaves: list[str],
                        lag_dias: int = MATURITY_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """Soma e contagem de `y` sobre o mesmo grupo até `t - lag_dias`.

    Substitui `lagged_outcome_sums`, e é aqui que o ganho é maior. Aquela função
    agregava por dia, tirava dois `cumsum`, fazia um `merge_asof` para trás e
    então precisava **guardar o rótulo da linha antes do merge** porque o
    `merge_asof` reinicia o índice — o H7, que deixou 94.145 pedidos sem
    histórico e deu a 120.060 o histórico de outra pessoa.

    Nada disso sobra. A amostragem em `t - lag` é a junção, e o resultado já vem
    preso à consulta que o gerou: não há índice a recasar, logo não há como
    recasar errado.
    """
    n = len(sub)
    ts = _segundos(sub["_reg"])
    indices = {k: _texto(sub[k]) for k in chaves}
    rid = np.arange(n, dtype="float64")

    fonte = tp.event_set(timestamps=ts,
                         features={"y": sub["y"].to_numpy("float64"), **indices},
                         indexes=chaves)
    # INVARIANTE: a mesma defasagem de maturação, de novo à mão. São dois
    # pontos porque são duas famílias de variável; esquecer em um só deixaria
    # metade do histórico maduro e metade não, sem erro visível.
    consulta = tp.event_set(timestamps=ts - lag_dias * SEGUNDOS_POR_DIA,
                            features={"rid": rid, **indices}, indexes=chaves)

    # INVARIANTE: `JANELA_TOTAL` faz o papel de "desde o começo". O Temporian
    # não tem janela infinita, então o acumulado é na verdade uma janela de 200
    # anos. Numa coorte que ultrapassasse isso, a soma truncaria em silêncio.
    soma = fonte["y"].moving_sum(window_length=JANELA_TOTAL, sampling=consulta)
    cont = fonte["y"].moving_count(window_length=JANELA_TOTAL, sampling=consulta)
    tabela = _para_pandas(tp.glue(soma.rename("cum_y"),
                                  cont.rename("cum_n").cast(tp.float64),
                                  consulta["rid"]))
    pronto = _na_ordem_das_linhas(tabela, ["cum_y", "cum_n"], n)
    return pronto["cum_y"], pronto["cum_n"]


def contagens_do_solicitante(sub: pd.DataFrame) -> dict[str, np.ndarray]:
    """Quantos pedidos o cidadão já fez, e há quanto tempo — sem tocar em rótulo.

    Substitui o bloco de `cumcount`/`diff`/`duplicated`. Estas variáveis não
    dependem de desfecho: que o cidadão já protocolou antes, inclusive hoje mais
    cedo, é fato conhecido na chegada, e por isso não levam defasagem.

    **Aqui a promessa do Temporian não ajuda, e é justo registrar.** Toda janela
    inclui o próprio evento, então cada contagem sai com um a mais e o `- 1`
    corrigindo à mão. Trocou-se um `cumcount` por um `moving_count` seguido de
    subtração: o número de pontos frágeis é o mesmo.

    O relógio destas contagens é o **intradiário** (ver `PASSO_DESEMPATE`), sem
    o qual os pedidos do mesmo dia se veriam uns aos outros em bloco.
    """
    n = len(sub)
    posicao = np.arange(n, dtype="float64")
    # INVARIANTE: relógio intradiário inventado aqui. `DataRegistro` é data sem
    # hora e o Temporian faz eventos simultâneos enxergarem uns aos outros;
    # sem este passo, cada pedido veria em bloco todos os do mesmo dia — o
    # vazamento de mesmo dia, 159.320 linhas na medição da auditoria externa.
    # Depende de `sub` já vir ordenado por (`_reg`, `IdPedido`).
    ts = _segundos(sub["_reg"]) + posicao * PASSO_DESEMPATE
    solicitante = _texto(sub["IdSolicitante"])
    orgao = _texto(sub["OrgaoDestinatario"])

    por_sol = tp.event_set(timestamps=ts,
                           features={"rid": posicao, "IdSolicitante": solicitante},
                           indexes=["IdSolicitante"])
    # INVARIANTE: `- 1` para excluir o próprio pedido da contagem. A janela
    # `(t-w, t]` inclui `t`, e para o Temporian isso está certo: o pedido de
    # agora aconteceu agora. Que ele não deva contar a si mesmo é decisão de
    # modelagem, e a biblioteca não tem como sustentá-la.
    n_previos = por_sol["rid"].moving_count(window_length=JANELA_TOTAL).cast(tp.float64) - 1.0
    # `since_last` devolve segundos; o original conta dias inteiros.
    desde_ultimo = por_sol["rid"].since_last() / SEGUNDOS_POR_DIA

    por_par = tp.event_set(
        timestamps=ts,
        features={"rid": posicao, "IdSolicitante": solicitante, "OrgaoDestinatario": orgao},
        indexes=["IdSolicitante", "OrgaoDestinatario"])
    cont_par = por_par["rid"].moving_count(window_length=JANELA_TOTAL).cast(tp.float64)
    # INVARIANTE: mesma exclusão do próprio evento, no par solicitante×órgão.
    n_previos_orgao = cont_par - 1.0
    # Primeira vez que este cidadão escreve a este órgão: a contagem acumulada,
    # que inclui o próprio evento, vale exatamente 1. `equal` e não `==`: o
    # `==` do Temporian compara os objetos, não os valores, e devolveria um
    # `bool` do Python em silêncio.
    primeiro_par = cont_par.equal(1.0).cast(tp.float64)

    # Para contar órgãos DISTINTOS já procurados, o mesmo evento precisa ser
    # lido por solicitante, não por par. `drop_index` reagrupa sem recalcular
    # nada — é a operação que no pandas exigia um segundo `groupby` e um
    # `cumsum` alinhado à mão.
    marcas = tp.glue(primeiro_par.rename("primeiro"), por_par["rid"])
    remarcado = marcas.drop_index("OrgaoDestinatario", keep=False)
    acumulado = remarcado["primeiro"].moving_sum(window_length=JANELA_TOTAL)
    # INVARIANTE: terceira exclusão do próprio evento. O acumulado inclui a
    # marca da linha corrente, então um órgão procurado pela primeira vez AGORA
    # já contaria como "distinto anterior" se esta subtração faltasse.
    distintos = acumulado - remarcado["primeiro"]

    tabela_sol = _para_pandas(tp.glue(n_previos.rename("n_pedidos_previos"),
                                      desde_ultimo.rename("dias_desde_ultimo_pedido"),
                                      por_sol["rid"]))
    tabela_par = _para_pandas(tp.glue(
        n_previos_orgao.rename("n_pedidos_previos_neste_orgao"), por_par["rid"]))
    tabela_dist = _para_pandas(tp.glue(
        distintos.rename("n_orgaos_distintos_previos"), remarcado["rid"]))

    saida: dict[str, np.ndarray] = {}
    saida.update(_na_ordem_das_linhas(
        tabela_sol, ["n_pedidos_previos", "dias_desde_ultimo_pedido"], n))
    saida.update(_na_ordem_das_linhas(
        tabela_par, ["n_pedidos_previos_neste_orgao"], n))
    saida.update(_na_ordem_das_linhas(
        tabela_dist, ["n_orgaos_distintos_previos"], n))
    # INVARIANTE: reposição do ausente no primeiro pedido de cada cidadão.
    # `since_last` não tem valor ali, e o zero com que `_na_ordem_das_linhas`
    # preenche faria "nunca pediu antes" virar "pediu hoje" — um estreante
    # indistinguível de um reincidente do mesmo dia.
    saida["dias_desde_ultimo_pedido"] = np.where(
        saida["n_pedidos_previos"] > 0, np.floor(saida["dias_desde_ultimo_pedido"]), np.nan)
    return saida


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
    # dia. É dela que o relógio intradiário de `contagens_do_solicitante`
    # deriva; trocar a chave de ordenação muda todas as contagens sem que nada
    # acuse. Verificado: correlação +0,996 com a data, 22 inversões em 655.177.
    df["_idp"] = pd.to_numeric(df.IdPedido, errors="coerce")
    df = df.sort_values(["_reg", "_idp"], kind="stable").reset_index(drop=True)

    # --- histórico do solicitante --------------------------------------------
    # Solicitante anonimizado ('0') não acumula: não é uma pessoa.
    real = df.IdSolicitante.ne("0")
    sub = df.loc[real]

    for col, v in contagens_do_solicitante(sub).items():
        df[col] = np.nan
        df.loc[real, col] = v.astype("float32")

    for col, chaves in (("prev_reenc_solicitante", ["IdSolicitante"]),
                        ("prev_reenc_neste_orgao",
                         ["IdSolicitante", "OrgaoDestinatario"])):
        cum_y, cum_n = desfechos_defasados(sub, chaves)
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
    roll = taxas_moveis_por_orgao(df)
    # H8: o prior tem de vir SÓ dos anos de treino. Com o prior de todo o
    # período, rótulo de validação e de teste entrava numa variável de entrada.
    base_prior = float(df.loc[df.ano.isin(TRAIN_YEARS), "y"].mean())
    for w in (90, 365):
        cnt, sm = roll[f"cnt_{w}"], roll[f"sum_{w}"]
        # Sem suavização, um órgão com um único pedido na janela produz taxa
        # 0,0 ou 1,0: 16,2% dos órgãos oscilavam mais de 5 pp por ruído de
        # volume baixo.
        df[f"orgao_rate_movel_{w}d"] = (sm + PRIOR_MOVEL * base_prior) / (cnt + PRIOR_MOVEL)
    nascimento = df.OrgaoDestinatario.map(births)
    df["dias_desde_primeiro_pedido_do_orgao"] = (
        df._reg - nascimento).dt.days.astype("float32")
    return df
