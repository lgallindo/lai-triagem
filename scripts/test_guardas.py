"""
Prova que as cinco guardas realmente pegam o que dizem pegar.

Uma guarda que passa sempre é pior que nenhuma, porque dá falsa segurança. Aqui
cada regra é provada por **mutação**: o defeito é injetado de propósito e a
guarda tem de acusar. Se ela passar, o teste falha.

    uv run python scripts/test_guardas.py

## Por que este arquivo existe, e não um comando ad hoc

Em 18/09/2026 eu testei uma guarda à mão, injetando o defeito com `sed -i` e
desfazendo com `git checkout -- README.md`. O `git checkout` não desfaz "a minha
mutação": ele desfaz **tudo** o que não está commitado — e levou consigo nove
correções de documentação que eu ainda não havia commitado. Trabalho perdido por
sequenciamento, não por engano de lógica.

As três salvaguardas embutidas aqui, e a razão de cada uma:

1. **Restauração por cópia byte a byte, nunca por `git`.** Antes de mutar, o
   arquivo é copiado para um temporário; a restauração devolve exatamente
   aquele conteúdo. Só desfaz a mutação, e é incapaz de tocar em trabalho
   alheio, commitado ou não.
2. **Recusa de arquivo sujo.** Se o alvo tem alteração não commitada, o teste
   aborta antes de mutar nada. Assim o teste nunca é a causa de perda.
3. **Conferência de identidade após restaurar.** Não basta restaurar: é preciso
   provar que restaurou, comparando byte a byte. Restauração silenciosamente
   incompleta seria o mesmo defeito com outra cara.

A quarta salvaguarda não é código: **este arquivo é o único caminho sancionado
para mutar**. A guarda de 18/09 existia dentro dos scripts; o que falhou foi eu
escrever o teste inline e contornar a minha própria proteção.
"""

import filecmp
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def guarda_passa(script):
    r = subprocess.run([sys.executable, "-m", "uv", "run", "python", script],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0, r.stdout


def roda_guarda(script):
    """Roda uma guarda com `uv run` e devolve (passou, saida)."""
    r = subprocess.run(["uv", "run", "python", script], cwd=ROOT,
                       capture_output=True, text=True)
    return r.returncode == 0, r.stdout + r.stderr


# ---------------------------------------------------------------- mutações
def texto(antigo, novo):
    """Mutação por substituição de texto."""
    def aplica(conteudo):
        if antigo not in conteudo:
            raise AssertionError(f"alvo da mutação não encontrado: {antigo[:60]!r}")
        return conteudo.replace(antigo, novo, 1)
    return aplica


def acrescenta(sufixo):
    return lambda conteudo: conteudo + sufixo


def json_mutante(fn):
    """Mutação sobre o JSON do artefato, que é uma única linha enorme."""
    def aplica(conteudo):
        d = json.loads(conteudo)
        fn(d)
        return json.dumps(d, ensure_ascii=False)
    return aplica


def _permuta(d):
    d["feature_order"][0], d["feature_order"][1] = \
        d["feature_order"][1], d["feature_order"][0]


def _email(d):
    d["caller_supplied_rationale"] = "contato: fulano.silva@orgao.gov.br"


def _post_hoc(d):
    d["feature_order"].append("AssuntoPedido")


CASOS = [
    # (rótulo, arquivo, mutação, guarda que deve acusar)
    ("número em prosa obsoleto", "README.md",
     texto("0,372225", "0,999999"), "scripts/check_prosa.py"),
    ("classificação de risco invertida no JSON", "README.md",
     texto('"alerta": "ALTO RISCO"', '"alerta": "BAIXO RISCO"'),
     "scripts/check_prosa.py"),
    ("histórico parcial no curl (7 de 8 campos)", "README.md",
     texto('"prev_reenc_solicitante_den":80,', ""), "scripts/check_prosa.py"),
    ("campo derivado enviado pelo chamador", "README.md",
     texto('"prev_reenc_solicitante":3,',
           '"prev_reenc_solicitante":3,"prev_reenc_rate_solicitante":0.0375,'),
     "scripts/check_prosa.py"),
    ("GET num endpoint que só aceita POST", "README.md",
     texto("## `POST /health`", "## `GET /health`"), "scripts/check_prosa.py"),
    ("ganho de variável divergente do booster", "README.md",
     texto("| `Municipio_sol` | Solicitantes | categórico | **8,26%** |",
           "| `Municipio_sol` | Solicitantes | categórico | **9,99%** |"),
     "scripts/check_prosa.py"),
    # O padrão `limiar <número>` não aparece em nenhum documento vigente -- só
    # em registro de auditoria, que é isento. A regra existe para impedir que o
    # número volte a ser escrito à mão, então a mutação acrescenta uma linha
    # nova a um documento vigente, que é como a reincidência aconteceria.
    ("limiar obsoleto reintroduzido na documentação", "docs/VERIFICATION.md",
     acrescenta("\nO limiar **0,999999** é o ponto de operação da fila.\n"),
     "scripts/check_docs_numbers.py"),
    ("número em prosa que o serviço não produz", "README.md",
     texto("0,080268", "0,070707"), "scripts/check_prosa.py"),
    ("feature_order permutado", "artifacts/preprocessor.json",
     json_mutante(_permuta), "scripts/check_artefato.py"),
    ("e-mail embarcado no artefato", "artifacts/preprocessor.json",
     json_mutante(_email), "scripts/check_artefato.py"),
    ("campo post hoc como variável", "artifacts/preprocessor.json",
     json_mutante(_post_hoc), "scripts/check_artefato.py"),
    ("comentário em inglês", "scripts/test_metrics.py",
     acrescenta("\n# This comment is in English, which the project rules do not\n"
                "# allow for comments, and the language check should detect it.\n"),
     "scripts/check_comentarios.py"),
    ("comentário citando caminho morto", "scripts/test_metrics.py",
     acrescenta("\n# Ver docs/ARQUIVO_QUE_NAO_EXISTE.md para os detalhes.\n"),
     "scripts/check_comentarios.py"),
]


def main():
    print(f"HEAD {git('rev-parse', '--short', 'HEAD')}\n")

    # Salvaguarda 2, aplicada de saída: nada de mutar com a árvore suja.
    sujo = git("status", "--porcelain")
    if sujo:
        print("ABORTADO — há alteração não commitada. Este teste muta arquivos\n"
              "do repositório e, embora restaure por cópia, não roda com a\n"
              "árvore suja: se algo der errado no meio, o que se perde é o seu\n"
              "trabalho. Commite ou guarde antes.\n")
        print(sujo)
        return 2

    falhas = []
    tmp = Path(tempfile.mkdtemp(prefix="mutacao-"))
    try:
        for rotulo, arquivo, mutacao, guarda in CASOS:
            alvo = ROOT / arquivo
            reserva = tmp / arquivo.replace("/", "__")

            # Salvaguarda 1: cópia byte a byte ANTES de qualquer escrita.
            shutil.copy2(alvo, reserva)
            try:
                original = alvo.read_text(encoding="utf-8")
                alvo.write_text(mutacao(original), encoding="utf-8")
                acusou, saida = roda_guarda(guarda)
                acusou = not acusou            # a guarda deve FALHAR
                if acusou:
                    linha = next((l.strip() for l in saida.splitlines()
                                  if "✗" in l), "")
                    print(f"  OK    {rotulo}")
                    if linha:
                        print(f"        {linha[:110]}")
                else:
                    print(f"  FALHA {rotulo}\n"
                          f"        {guarda} PASSOU com o defeito injetado")
                    falhas.append(rotulo)
            except AssertionError as e:
                print(f"  ERRO  {rotulo}: {e}")
                falhas.append(rotulo)
            finally:
                # Salvaguarda 1 (restauração) e 3 (conferência).
                shutil.copy2(reserva, alvo)
                if not filecmp.cmp(reserva, alvo, shallow=False):
                    print(f"  !! restauração de {arquivo} não confere byte a byte")
                    falhas.append(f"restauração de {arquivo}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    sujo_depois = git("status", "--porcelain")
    if sujo_depois:
        print("!! a árvore ficou suja depois do teste:")
        print(sujo_depois)
        falhas.append("árvore suja ao final")
    else:
        print("árvore limpa ao final: a restauração por cópia devolveu tudo")

    for g in ("scripts/test_metrics.py", "scripts/check_docs_numbers.py",
              "scripts/check_prosa.py", "scripts/check_artefato.py",
              "scripts/check_comentarios.py"):
        passou, _ = roda_guarda(g)
        if not passou:
            print(f"!! {g} falha no repositório intacto")
            falhas.append(f"{g} no repo intacto")

    print()
    if falhas:
        print(f"FALHOU: {len(falhas)} caso(s) — {falhas}")
        return 1
    print(f"OK — as {len(CASOS)} regras acusaram o defeito injetado, e o "
          f"repositório voltou ao estado original")
    return 0


if __name__ == "__main__":
    sys.exit(main())
