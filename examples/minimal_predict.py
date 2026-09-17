"""EXEMPLO MÍNIMO — pontua um pedido LAI. Não requer BentoML.

    cd ~/lai-triagem && uv run python examples/minimal_predict.py

Carrega o modelo em texto nativo do LightGBM mais o acompanhante JSON, pontua um
pedido com dados de chegada e imprime o sinalizador ALTO/BAIXO RISCO. É
exatamente o caminho de código que o serviço BentoML envolve: se isto funciona,
o serviço também funciona.

Demonstra os três modos de operação:
  1. com histórico do solicitante informado pelo chamador  -> precisão plena
  2. sem histórico                                          -> degradado
  3. com campo posterior à triagem                          -> recusado
"""

import sys
from pathlib import Path

import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lai_triagem.featurize import Preprocessor, risk_label  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"

booster = lgb.Booster(model_file=str(ART / "model_arrival.txt"))
prep = Preprocessor.from_json(ART / "preprocessor.json")

# Tudo aqui é conhecido no momento em que o pedido entra na caixa do Fala.BR.
pedido = {
    "OrgaoDestinatario": "CC-PR – Casa Civil da Presidência da República",
    "Esfera": "Federal",
    "UF": " ",
    "Municipio": " ",
    "FormaResposta": "Pelo sistema (com avisos por email)",
    "OrigemSolicitacao": "Internet",
    "DataRegistro": "15/09/2026",
    "TipoDemandante": "Pessoa Física",
    "Genero": "Masculino",
    "Escolaridade": "Ensino Fundamental",
    "Profissao": "Outra",
    "Pais": "Brasil",
    "UF_sol": "PE",
    "Municipio_sol": "Recife",
    "DataNascimento": "08/10/1976",
}

# Histórico do solicitante: dado pessoal, fornecido pelo CHAMADOR, nunca
# embarcado no artefato. Ver docs/DECISAO_ESTADO_SOLICITANTE.md.
historico_estreante = {
    "n_pedidos_previos": 0,
    "prev_reenc_solicitante": 0,
    "prev_reenc_rate_solicitante": -1,   # indefinida sem pedido anterior
    "n_pedidos_previos_neste_orgao": 0,
    "prev_reenc_neste_orgao": 0,
    "n_orgaos_distintos_previos": 0,
    "dias_desde_ultimo_pedido": -1,
}
historico_veterano = {
    "n_pedidos_previos": 80,
    "prev_reenc_solicitante": 3,
    "prev_reenc_rate_solicitante": 3 / 80,
    "n_pedidos_previos_neste_orgao": 12,
    "prev_reenc_neste_orgao": 0,
    "n_orgaos_distintos_previos": 14,
    "dias_desde_ultimo_pedido": 5,
}


def pontuar(rotulo, extra=None):
    payload = dict(pedido, **(extra or {}))
    X = prep.transform(payload)
    prob = float(booster.predict(X)[0])
    print(f"  {rotulo:<34} P={prob:.4f}  {risk_label(prob, prep.threshold)}"
          f"   histórico informado: {prep.history_supplied(payload)}")
    return prob


print(f"limiar da fila: {prep.threshold:.6f}   "
      f"({100 * prep.meta['queue_fraction']:.0f}% mais arriscados)")
print(f"órgão: {pedido['OrgaoDestinatario']}")
print(f"  taxa histórica {prep.organ_rate.get(pedido['OrgaoDestinatario']):.4f}"
      f"   móvel 90d {prep.organ_movel_90.get(pedido['OrgaoDestinatario']):.4f}"
      f"   (taxa-base da coorte {prep.base_rate:.4f})\n")

print("1. modos de operação do histórico do solicitante")
pontuar("estreante (0 pedidos)", historico_estreante)
pontuar("veterano (80 pedidos, 12 aqui)", historico_veterano)
pontuar("sem histórico informado", None)

print("\n2. contraste de órgão — competência estreita pontua muito mais baixo")
for orgao in ("UFLA – Universidade Federal de Lavras",
              "SGPR – Secretaria-Geral da Presidência da República"):
    payload = dict(pedido, OrgaoDestinatario=orgao, **historico_estreante)
    prob = float(booster.predict(prep.transform(payload))[0])
    print(f"  {orgao[:46]:<46} P={prob:.4f}  {risk_label(prob, prep.threshold)}")

print("\n3. barreira de vazamento — campo posterior à triagem deve ser recusado")
for campo, valor in (("FoiProrrogado", "Sim"), ("AssuntoPedido", "Recursos Humanos"),
                     ("PrazoAtendimento", "05/10/2026")):
    try:
        prep.transform(dict(pedido, **{campo: valor}))
        print(f"  !! {campo} NÃO foi recusado")
    except ValueError:
        print(f"  OK  {campo} recusado")
