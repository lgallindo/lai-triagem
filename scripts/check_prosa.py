"""
Camada de auditoria #13 — a prosa da documentação contra a realidade.

Por que esta camada existe: em 18/09/2026 o `check_docs_numbers.py` passou com
código zero enquanto o exemplo principal do README devolvia a classificação de
risco **oposta** à documentada. Aquele guarda confere contagens e o limiar; o
limiar exibido estava correto, e `0,076869` não era grandeza guardada. Código de
saída zero significa "nada estourou", não "a documentação é verdadeira".

O que se confere aqui, e cada item corresponde a um defeito real já ocorrido:

  1. Todo payload de `curl` documentado manda o conjunto COMPLETO dos oito
     campos de histórico, ou nenhum. Nunca um subconjunto -- o contrato é
     atômico, e mandar sete faz o serviço descartar os oito em silêncio.
  2. Todo bloco ```json que segue um `curl` bate, campo a campo, com o que o
     modelo de fato devolve para aquele payload. A pontuação é feita em
     processo, pelo mesmo caminho de código que o serviço embrulha, para não
     depender de servidor de pé nem de porta livre.
  3. Nenhum documento afirma em prosa algo que o artefato contradiz -- por
     exemplo "todas opcionais" sob um contrato atômico, ou "sem demografia"
     quando as demográficas estão no modelo.
  4. Nenhum documento promete `GET` num endpoint que só aceita `POST`.

    uv run python scripts/check_prosa.py
"""

import json
import re
import sys
from pathlib import Path

import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lai_triagem.featurize import Preprocessor, risk_label  # noqa: E402

ART = ROOT / "artifacts"
prep = Preprocessor.from_json(ART / "preprocessor.json")
booster = lgb.Booster(model_file=str(ART / "model_arrival.txt"))

CAMPOS_CHAMADOR = set(prep.caller_features)
DOCS = sorted(list(ROOT.glob("*.md")) + list(ROOT.glob("docs/**/*.md")))

# Os registros de auditoria CITAM os defeitos que encontraram, textualmente.
# Conferi-los procurando esses mesmos defeitos é erro de categoria: a primeira
# execução deste guarda acusou cinco divergências, e todas as cinco eram linhas
# relatando um defeito, não cometendo-o. A isenção é do diretório inteiro, e é
# impressa em toda execução -- omissão silenciosa é pior que isenção declarada.
HISTORICOS = [d for d in DOCS if "auditorias" in d.parts]
VIGENTES = [d for d in DOCS if d not in HISTORICOS]

falhas: list[str] = []
avisos: list[str] = []


def pontua_baseline(payload: dict) -> dict:
    """Reproduz `/score_baseline`: a consulta por órgão, sem modelo."""
    taxa = float(prep.organ_rate.get(payload["OrgaoDestinatario"], prep.base_rate))
    return {
        "probabilidade_reencaminhamento": round(taxa, 6),
        "alerta": risk_label(taxa, prep.threshold),
        "metodo": "lookup histórico por órgão (sem modelo)",
        "orgao_conhecido": prep.organ_is_known(payload["OrgaoDestinatario"]),
    }


def pontua(payload: dict) -> dict:
    """Reproduz `/score` em processo, campo a campo igual ao service.py."""
    limpo = {k: v for k, v in payload.items() if v is not None}
    X = prep.transform(limpo)
    prob = float(booster.predict(X)[0])
    cal = prep.calibrate(prob)
    return {
        "probabilidade_reencaminhamento": round(prob, 6),
        "alerta": risk_label(prob, prep.threshold),
        "threshold": prep.threshold,
        "probabilidade_calibrada": None if cal is None else round(cal, 6),
        "calibrada_apenas_para_leitura": True,
        "orgao_conhecido": prep.organ_is_known(limpo["OrgaoDestinatario"]),
        "orgao_rate_historica": round(float(X["orgao_rate"].iloc[0]), 6),
        "orgao_rate_movel_90d": round(float(X["orgao_rate_movel_90d"].iloc[0]), 6),
        "base_rate_coorte": round(prep.base_rate, 6),
        "historico_informado": prep.history_supplied(limpo),
        "historico_parcial_ignorado": prep.history_partial_ignored(limpo),
    }


# ---------------------------------------------------------------------------
# 1 e 2 — os `curl` documentados, e a resposta que eles de fato produzem
# ---------------------------------------------------------------------------
RX_CURL = re.compile(r"curl[^\n]*?-d '(\{.*?\})'", re.S)
RX_JSON = re.compile(r"```json\n(.*?)\n```", re.S)

n_curls = n_comparados = 0
# Valores que o serviço de fato produz, e por isso podem ser citados em prosa.
# Serve à varredura global mais abaixo: qualquer número de seis decimais escrito
# com VÍRGULA é prosa (o JSON usa ponto), então tem de ser um destes.
legitimos: set[float] = {prep.threshold, prep.base_rate}
for doc in VIGENTES:
    texto = doc.read_text(encoding="utf-8")
    for m in RX_CURL.finditer(texto):
        bruto = m.group(1)
        try:
            corpo = json.loads(bruto)
        except json.JSONDecodeError as e:
            falhas.append(f"{doc.relative_to(ROOT)}  payload de curl não é JSON válido: {e}")
            continue
        if "pedido" not in corpo:
            continue
        pedido = corpo["pedido"]
        n_curls += 1
        rotulo = f"{doc.relative_to(ROOT)}  curl #{n_curls}"

        # (1) o conjunto de histórico é atômico: ou os oito, ou nenhum.
        presentes = CAMPOS_CHAMADOR & set(pedido)
        if presentes and presentes != CAMPOS_CHAMADOR:
            faltam = sorted(CAMPOS_CHAMADOR - presentes)
            falhas.append(
                f"{rotulo}: manda {len(presentes)} de {len(CAMPOS_CHAMADOR)} campos de "
                f"histórico; faltam {faltam}. O contrato é atômico: o serviço "
                f"descartaria os {len(presentes)} em silêncio.")

        # Campos que parecem de histórico mas não estão no contrato.
        derivados = set(prep.meta.get("derived_from_caller", []))
        intrusos = derivados & set(pedido)
        if intrusos:
            falhas.append(
                f"{rotulo}: manda {sorted(intrusos)}, que o serviço deriva "
                f"internamente e ignora na entrada.")

        # (2b) Localidade: o trecho entre este `curl` e o próximo tem de citar,
        # em algum lugar, o escore que ESTE payload produz. Antes isto olhava só
        # os 700 caracteres seguintes e cortava no primeiro ```json -- e quando
        # a resposta completa passou a ser documentada, a frase em prosa foi
        # empurrada para depois do bloco e saiu da janela. O teste de mutação
        # pegou a regressão: a regra passou a aceitar 0,999999 em silêncio.
        proximo = RX_CURL.search(texto, m.end())
        regiao = texto[m.end():proximo.start() if proximo else len(texto)]
        if "/score_baseline" in m.group(0):
            esperado = round(float(prep.organ_rate.get(
                pedido["OrgaoDestinatario"], prep.base_rate)), 6)
        else:
            esperado = pontua(pedido)["probabilidade_reencaminhamento"]
        # Comparação NUMÉRICA, não textual: `json.dumps` grava `0.31659` e o
        # formato de seis casas produz `0.316590`. Os dois são o mesmo número,
        # e a primeira versão desta regra reprovava a diferença de zero à
        # direita — divergência entre duas ferramentas minhas, não da prosa.
        citados = [float(t.replace(",", "."))
                   for t in re.findall(r"\b\d+[.,]\d{3,6}\b", regiao)]
        if not any(abs(c - esperado) < 1e-6 for c in citados):
            falhas.append(
                f"{rotulo}: nem o bloco de resposta nem a prosa até o próximo "
                f"comando citam {esperado:.6f}, que é o que este payload produz")

        # (2) o bloco ```json seguinte tem de bater com a resposta real.
        seguinte = RX_JSON.search(texto[m.end():])
        if not seguinte:
            continue
        try:
            documentado = json.loads(seguinte.group(1))
        except json.JSONDecodeError:
            continue
        if "probabilidade_reencaminhamento" not in documentado:
            continue

        n_comparados += 1
        try:
            # Cada endpoint tem contrato próprio: comparar a resposta da linha
            # de base contra o escore do modelo daria divergência falsa.
            real = (pontua_baseline(pedido) if "/score_baseline" in m.group(0)
                    else pontua(pedido))
            # Todo valor que o serviço de fato devolve é legítimo em prosa.
            legitimos.update(v for v in real.values() if isinstance(v, float))
        except Exception as e:  # payload documentado que nem pontua é defeito
            falhas.append(f"{rotulo}: o payload documentado não pontua: {e!r}")
            continue
        for campo, valor_doc in documentado.items():
            if campo not in real:
                avisos.append(f"{rotulo}: documenta `{campo}`, que /score não devolve")
                continue
            valor_real = real[campo]
            if isinstance(valor_doc, float) and isinstance(valor_real, float):
                bate = abs(valor_doc - valor_real) < 1e-6
            else:
                bate = valor_doc == valor_real
            if not bate:
                falhas.append(
                    f"{rotulo}: campo `{campo}` documentado como {valor_doc!r}, "
                    f"real {valor_real!r}")

# ---------------------------------------------------------------------------
# 2d — todo número de seis decimais escrito em PROSA tem de ser um valor que o
# serviço realmente produz. O discriminador é a pontuação: o JSON usa ponto
# decimal, a prosa em português usa vírgula. Sem depender de janela nenhuma,
# isto pega número obsoleto em qualquer lugar do texto.
# ---------------------------------------------------------------------------
RX_PROSA_DECIMAL = re.compile(r"\b(\d+,\d{6})\b")
n_prosa = 0
for doc in VIGENTES:
    for n, linha in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
        for achado in RX_PROSA_DECIMAL.findall(linha):
            n_prosa += 1
            valor = float(achado.replace(",", "."))
            if not any(abs(valor - v) < 1e-6 for v in legitimos):
                falhas.append(
                    f"{doc.relative_to(ROOT)}:{n}  prosa cita {achado}, que não "
                    f"é nenhum valor que o serviço produz para os payloads "
                    f"documentados")

# ---------------------------------------------------------------------------
# 2c — ganho por variável citado em tabela contra o ganho medido no booster
#
# Estes percentuais são mantidos à mão em três tabelas e já divergiram QUATRO
# vezes; a auditoria da camada 3 achou duas instâncias que eu não tinha visto,
# com o README se contradizendo internamente (20,19% na conclusão contra 11,56%
# na tabela). Isto fecha a classe.
# ---------------------------------------------------------------------------
ganhos_brutos = booster.feature_importance(importance_type="gain")
total_ganho = float(ganhos_brutos.sum())
GANHO = {n: 100.0 * g / total_ganho
         for n, g in zip(booster.feature_name(), ganhos_brutos, strict=True)}

RX_LINHA_TABELA = re.compile(r"^\|\s*`([^`]+)`")
RX_PCT = re.compile(r"(\d+,\d+)%")
# Só vale dentro de tabela cujo CABEÇALHO fala de ganho. Sem isso a regra
# confunde percentual de ausência com percentual de ganho: a tabela H4 de
# VERIFICATION.md tem cabeçalho `| Campo | Ausente | Distintos |` e diz que
# `Escolaridade` está 76,23% ausente, o que não é ganho nenhum.
n_ganhos = 0
for doc in VIGENTES:
    em_tabela_de_ganho = False
    for n, linha in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
        if not linha.startswith("|"):
            em_tabela_de_ganho = False
            continue
        if "Ganho" in linha or "Uso no modelo" in linha:
            em_tabela_de_ganho = True
            continue
        if not em_tabela_de_ganho:
            continue
        achado_nome = RX_LINHA_TABELA.match(linha)
        if not achado_nome or achado_nome.group(1) not in GANHO:
            continue
        nome = achado_nome.group(1)
        # "baixo" ocupa a coluna de ganho em algumas linhas; aí o percentual
        # que aparece na linha é outra coisa.
        if "baixo" in linha or "ausente" in linha:
            continue
        pcts = RX_PCT.findall(linha)
        if not pcts:
            continue
        n_ganhos += 1
        primeiro = float(pcts[0].replace(",", "."))
        esperado = round(GANHO[nome], 2)
        if abs(primeiro - esperado) > 0.01:
            falhas.append(
                f"{doc.relative_to(ROOT)}:{n}  ganho de `{nome}`: tabela diz "
                f"{primeiro:.2f}%, booster diz {esperado:.2f}%")

# ---------------------------------------------------------------------------
# 3 — afirmações em prosa que o artefato contradiz
# ---------------------------------------------------------------------------
tem_demografia = any(
    c in prep.feature_order
    for c in ("Escolaridade", "Profissao", "Genero", "idade"))
atomico = bool(prep.meta.get("caller_supplied_atomic"))

# (regex, condição para ser defeito, explicação)
PROIBIDAS = [
    (re.compile(r"[Tt]odas opcionais"), atomico,
     "diz 'todas opcionais' mas o contrato do chamador é atômico"),
    (re.compile(r"[Ss]em dado demogr[áa]fico algum|n[ãa]o usa.{0,20}demogr[áa]fic"),
     tem_demografia,
     "nega o uso de demografia, mas há variáveis demográficas no modelo"),
    (re.compile(r"corre[çc][ãa]o pendente"), True,
     "anuncia correção pendente; confirme que não foi já corrigida"),
]

for doc in VIGENTES:
    for n, linha in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
        if any(m in linha for m in ("anteriores ao Fix", "já foi muito melhor",
                                    "ersões anteriores", "antes do Fix",
                                    "era vazamento", "Registro histórico")):
            continue
        for rx, condicao, porque in PROIBIDAS:
            if condicao and rx.search(linha):
                alvo = falhas if "pendente" not in porque else avisos
                alvo.append(f"{doc.relative_to(ROOT)}:{n}  {porque}: {linha.strip()[:90]}")

# ---------------------------------------------------------------------------
# 4 — método HTTP: todo endpoint `@bentoml.api` é POST
# ---------------------------------------------------------------------------
servico = (ROOT / "service.py").read_text(encoding="utf-8")
endpoints = re.findall(r"@bentoml\.api\s*\n\s*def\s+(\w+)", servico)
for doc in VIGENTES:
    for n, linha in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
        # Uma linha que cita o 405 está EXPLICANDO que GET falha, não
        # prometendo GET. Distinguir afirmação de citação é o ponto difícil de
        # auditar prosa, e é onde este guarda errou na primeira execução.
        if "405" in linha:
            continue
        for ep in endpoints:
            # `/healthz` é sonda do próprio BentoML e é GET de verdade.
            if re.search(rf"GET\s+`?/{ep}`?(?![a-z])", linha):
                falhas.append(
                    f"{doc.relative_to(ROOT)}:{n}  promete GET /{ep}, mas "
                    f"`@bentoml.api` só aceita POST (devolve HTTP 405)")

# ---------------------------------------------------------------------------
print(f"documentos vigentes varridos: {len(VIGENTES)}")
print(f"isentos por serem registro histórico: {len(HISTORICOS)} "
      f"({', '.join(d.name for d in HISTORICOS)})")
print(f"payloads de curl conferidos: {n_curls}")
print(f"respostas comparadas:        {n_comparados}")
print(f"ganhos de variável conferidos: {n_ganhos}")
print(f"decimais em prosa conferidos: {n_prosa}  (valores legítimos: {len(legitimos)})")
print(f"endpoints lidos do service:  {endpoints}")

if avisos:
    print(f"\n{len(avisos)} aviso(s), não bloqueiam:")
    for a in avisos:
        print(f"  ~ {a}")

if falhas:
    print(f"\nFALHOU: {len(falhas)} divergência(s) entre prosa e realidade")
    for f in falhas:
        print(f"  ✗ {f}")
    sys.exit(1)

print("\nOK — a prosa não contradiz o artefato nem o serviço")
