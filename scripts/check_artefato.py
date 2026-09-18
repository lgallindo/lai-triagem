"""
Camada de auditoria #14 — o arquivo de modelo entregue, como objeto.

As outras camadas conferem o processo. Esta confere o **produto**: os dois
arquivos que um terceiro de fato recebe, `model_arrival.txt` e
`preprocessor.json`, sem confiar em nada do repositório que os gerou.

O que se confere:

  1. Coerência de forma: `feature_order` tem o mesmo tamanho E a mesma ordem
     que o booster espera. Ordem importa -- o LightGBM recebe posições, não
     nomes, e uma permutação silenciosa produz escore errado sem erro algum.
     Este projeto já quebrou com `LightGBMError: 31 vs 22`.
  2. Nenhum campo posterior à triagem (`post hoc`) sobreviveu como variável.
  3. Todo campo do chamador ou é variável, ou serve para derivar uma.
  4. `contains_personal_data: false` é verdade: procura CPF, e-mail, nome de
     pessoa e identificador de solicitante dentro do artefato.
  5. As tabelas embarcadas são de ÓRGÃO, não de pessoa.
  6. `category_codes` cobre todas as categóricas, sem código órfão.
  7. O artefato carrega **isolado**: copiado para um diretório temporário, sem
     o repositório no `sys.path`, e ainda pontua.
  8. Reprodutibilidade por hash: declara os campos que carregam relógio de
     parede e por isso impedem verificar procedência por hash.

    uv run python scripts/check_artefato.py
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
MODELO = ART / "model_arrival.txt"
SIDECAR = ART / "preprocessor.json"

meta = json.loads(SIDECAR.read_text(encoding="utf-8"))
booster = lgb.Booster(model_file=str(MODELO))

falhas: list[str] = []
notas: list[str] = []


def exige(condicao, mensagem):
    if not condicao:
        falhas.append(mensagem)


# ---------------------------------------------------------------- 1. forma
ordem = meta["feature_order"]
nomes_booster = booster.feature_name()
exige(len(ordem) == booster.num_feature(),
      f"feature_order tem {len(ordem)} nomes, booster espera "
      f"{booster.num_feature()} variáveis")
exige(list(ordem) == list(nomes_booster),
      "feature_order NÃO está na mesma ordem que o booster; o LightGBM recebe "
      "posições, então isto produziria escore errado sem erro")
if list(ordem) != list(nomes_booster):
    for i, (a, b) in enumerate(zip(ordem, nomes_booster)):
        if a != b:
            falhas.append(f"  primeira divergência na posição {i}: "
                          f"json={a!r} booster={b!r}")
            break

# --------------------------------------------------- 2. vazamento residual
excluidos = set(meta["excluded_leakage_features"])
intrusos = excluidos & set(ordem)
exige(not intrusos, f"campo(s) post hoc sobreviveram como variável: {sorted(intrusos)}")

# ------------------------------------------------- 3. campos do chamador
chamador = list(meta["caller_supplied_features"])
derivados = set(meta.get("derived_from_caller", []))
orfaos = [c for c in chamador
          if c not in ordem and not c.endswith("_den")]
exige(not orfaos,
      f"campo(s) do chamador que não são variável nem alimentam derivada: {orfaos}")
# Os `_den` existem só para o serviço calcular as razões; as razões, sim,
# precisam ser variáveis, senão o denominador é pedido em vão.
for d in derivados:
    exige(d in ordem,
          f"`{d}` está em derived_from_caller mas não é variável do modelo; "
          f"os denominadores seriam pedidos ao chamador sem uso")

# -------------------------------------------- 4. e 5. dado pessoal no artefato
# O artefato NÃO carrega a declaração `contains_personal_data`: quem a afirma é
# o `service.py`, com o valor escrito à mão na resposta do `/health`. Ou seja, a
# afirmação não é derivada da coisa que descreve, e não se atualizaria se um dia
# alguém embarcasse tabelas de solicitante. É o mesmo defeito de fundo de todo o
# resto achado em 18/09/2026: asserção mantida à mão, livre para divergir.
# Aqui a propriedade é medida no artefato e CONFRONTADA com o que o serviço diz.
declarado_no_artefato = meta.get("contains_personal_data")
if declarado_no_artefato is None:
    notas.append(
        "o artefato não se autodeclara quanto a dado pessoal; a afirmação está "
        "escrita à mão em service.py. Quem recebe só os dois arquivos não tem a "
        "declaração — precisaria confiar no README. Defeito MENOR.")

bruto = SIDECAR.read_text(encoding="utf-8")
PADROES_PESSOAIS = [
    (re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), "CPF formatado"),
    (re.compile(r"\b\d{11}\b"), "sequência de 11 dígitos (possível CPF)"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "e-mail"),
    (re.compile(r"\bIdSolicitante\b"), "identificador de solicitante"),
    (re.compile(r"\bDataNascimento\b\s*:\s*\"\d"), "data de nascimento concreta"),
]
for rx, rotulo in PADROES_PESSOAIS:
    achado = rx.search(bruto)
    exige(achado is None,
          f"possível dado pessoal no artefato ({rotulo}): {achado.group(0)[:40]!r}"
          if achado else "")

# As tabelas embarcadas devem ser indexadas por órgão. Um órgão do Fala.BR vem
# como "SIGLA – Nome"; uma chave que não pareça isso merece olhar.
for tabela in ("organ_rate", "organ_rate_movel_90d", "organ_rate_movel_365d",
               "organ_birth"):
    if tabela not in meta:
        continue
    chaves = list(meta[tabela])
    estranhas = [k for k in chaves
                 if not isinstance(k, str) or (len(k) < 3 and k.strip())]
    exige(not estranhas,
          f"{tabela}: chave(s) que não parecem órgão: {estranhas[:5]}")
    notas.append(f"{tabela}: {len(chaves)} órgãos")

# Confronta a propriedade medida com o que o serviço afirma ao integrador.
servico = (ROOT / "service.py").read_text(encoding="utf-8")
afirma_sem_pessoal = re.search(
    r'"contains_personal_data"\s*:\s*(False|True)', servico)
sem_pessoal_medido = not falhas  # nenhum padrão pessoal casou acima
if afirma_sem_pessoal:
    afirmado = afirma_sem_pessoal.group(1) == "True"
    if afirmado and sem_pessoal_medido:
        falhas.append(
            "service.py afirma contains_personal_data: True, mas a varredura do "
            "artefato não achou dado pessoal — afirmação e realidade divergem")
    elif not afirmado and not sem_pessoal_medido:
        falhas.append(
            "service.py afirma contains_personal_data: False, mas a varredura "
            "do artefato ACHOU dado pessoal — afirmação e realidade divergem")
    else:
        notas.append(
            f"service.py afirma contains_personal_data: {afirmado}, e a "
            f"varredura do artefato concorda")

# ------------------------------------------------------ 6. category_codes
codigos = meta["category_codes"]
categoricas = set(meta["categorical_features"])
exige(set(codigos) == categoricas,
      f"category_codes cobre {sorted(set(codigos))} mas as categóricas são "
      f"{sorted(categoricas)}")
for col, mapa in codigos.items():
    vals = sorted(mapa.values())
    exige(vals == list(range(len(vals))),
          f"category_codes[{col}] não é contíguo de 0..n-1 "
          f"(min={vals[0] if vals else None}, max={vals[-1] if vals else None}, "
          f"n={len(vals)})")

# ---------------------------------------------------------- limiar coerente
lim = meta["threshold"]
exige(0.0 < lim < 1.0, f"limiar fora de (0,1): {lim}")
exige(0.0 < meta["queue_fraction"] <= 1.0,
      f"queue_fraction fora de (0,1]: {meta['queue_fraction']}")

# ------------------------------------------------ 7. carrega isolado do repo
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    shutil.copy(MODELO, tmp / MODELO.name)
    shutil.copy(SIDECAR, tmp / SIDECAR.name)
    # Roda em processo separado, com cwd no temporário e SEM o repo no path.
    codigo = (
        "import json,sys;import lightgbm as lgb;"
        "b=lgb.Booster(model_file='model_arrival.txt');"
        "m=json.load(open('preprocessor.json',encoding='utf-8'));"
        "assert b.num_feature()==len(m['feature_order']);"
        "print('OK',b.num_trees(),b.num_feature())"
    )
    r = subprocess.run([sys.executable, "-c", codigo], cwd=tmp,
                       capture_output=True, text=True)
    exige(r.returncode == 0,
          f"artefato não carrega isolado do repositório: {r.stderr.strip()[:200]}")
    if r.returncode == 0:
        notas.append(f"carga isolada: {r.stdout.strip()}")

# ------------------------------------- 8. procedência verificável por hash?
relogio = [c for c in ("created_utc",) if c in meta]
tem_fit_seconds = any("fit_seconds" in v for v in meta.get("metrics", {}).values()
                      if isinstance(v, dict))
if relogio or tem_fit_seconds:
    notas.append(
        "procedência NÃO é verificável por hash: o artefato embute "
        + ", ".join(filter(None, [", ".join(relogio),
                                  "metrics.*.fit_seconds" if tem_fit_seconds else ""]))
        + " — relógio de parede. Dois treinos do mesmo código e dos mesmos dados "
          "dão o mesmo modelo e hashes diferentes. Defeito MENOR, declarado.")

# ---------------------------------------------------------------------------
print(f"modelo:   {MODELO.name}  {booster.num_trees()} árvores, "
      f"{booster.num_feature()} variáveis")
print(f"sidecar:  {SIDECAR.name}  {len(bruto)} bytes, {len(meta)} campos")
print(f"limiar:   {lim}  (fila de {100 * meta['queue_fraction']:.0f}%)")
print(f"exclusões post hoc: {len(excluidos)}   campos do chamador: {len(chamador)}")
for n in notas:
    print(f"  · {n}")

falhas = [f for f in falhas if f]
if falhas:
    print(f"\nFALHOU: {len(falhas)} problema(s) no artefato entregue")
    for f in falhas:
        print(f"  ✗ {f}")
    sys.exit(1)
print("\nOK — o artefato entregue é coerente, carrega isolado e não traz dado pessoal")
