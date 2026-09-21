"""
Conta as invariantes temporais mantidas à mão na superfície de variáveis.

Esta é a medida que decide a comparação de `docs/LINHA_BASE_COMPLEXIDADE.md`.
Ela estava sendo contada a olho, e medida contada a olho não se compara entre
ramos: cada um contaria do seu jeito. Aqui a regra é executável.

## Duas contagens, e por que não bastava uma

A primeira versão deste script contava só o vocabulário do pandas. Aplicada ao
ramo do Temporian, devolveu **zero** — e zero era mentira. O módulo continuava
tendo defasagem escolhida à mão, exclusão do próprio evento feita com um
`- 1`, e um relógio intradiário inventado para desempatar. Nada disso é
`searchsorted`, e tudo isso é exatamente a mesma coisa: um ponto onde uma
pessoa tem de acertar sozinha, sem que nada verifique.

Uma medida que zera quando se troca de biblioteca não mede risco, mede
vocabulário. Então há duas contagens:

**(A) regra publicada** — os tokens que `LINHA_BASE_COMPLEXIDADE.md` nomeia.
Reproduz os **14** de `develop`, o que serve de aferição: a régua bate com a
contagem à mão que ela substitui.

**(B) invariantes declaradas** — cada ponto marcado no código com o comentário
`# INVARIANTE:`. Existe porque (A) só enxerga um vocabulário, e um ramo pode
zerá-la sem ter removido risco nenhum. A marca é obrigação de quem escreve, e é
`grep`-ável: quem revisa confere se falta alguma.

O total relatado é **(A) + (B)**. Em `develop` não há marca alguma, logo o
total é 14 e a linha de base publicada continua valendo sem retoque.

**A medida é auto-declarada, e isso é limitação, não detalhe.** Quem escreve
pode esquecer de marcar. A defesa é a mesma que o projeto usa em toda parte:
isenção declarada em vez de omissão silenciosa. A auditoria do ramo enumera as
marcas uma a uma, para que a conferência seja possível.

**A regra (A).** Conta-se, apenas em código executável (comentário e docstring
são descartados pelo `tokenize`, senão um comentário explicando um
`searchsorted` contaria como um `searchsorted`), cada ocorrência de:

  * `searchsorted`      — busca de fronteira posicional feita à mão
  * `side=`             — a escolha `left` contra `right` nessa fronteira
  * `cumsum` / `cumcount` — acumulação que depende da ordem das linhas
  * `shift`             — deslocamento de uma posição, à mão
  * `merge_asof`        — junção pelo "mais recente até"
  * `reindex` / `.index.to_numpy()` — preservação e recasamento de rótulo em
    volta de um `merge_asof`, que foi exatamente o H7

A lista sai de `docs/LINHA_BASE_COMPLEXIDADE.md`, que nomeia essas mesmas
construções. Aplicada a `develop` ela devolve **14**, que é o número publicado
como linha de base — a regra reproduz a contagem à mão, não a substitui por
outra.

O que a contagem NÃO afirma: que um número menor seja melhor. Ela mede quantos
pontos dependem de uma pessoa acertar "não olhe para o futuro" sem que nada
verifique. Um ramo que zere a contagem porque deixou de calcular a variável não
melhorou nada — por isso a regra de aceitação de desempenho vem junto.

    uv run python scripts/conta_invariantes.py
"""

import ast
import io
import sys
import tokenize
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Superfície medida: onde as variáveis são construídas. Declarada aqui, e não
# inferida, para que a mesma regra valha em ramos que organizam o código de
# outro jeito. `None` no lugar das funções significa "o arquivo inteiro".
SUPERFICIE: list[tuple[str, list[str] | None]] = [
    ("scripts/train.py", ["organ_birth_table", "organ_rolling",
                          "lagged_outcome_sums", "build_features"]),
    ("lai_triagem/codificacao.py", None),
    ("lai_triagem/dados.py", None),
]

# Superfícies alternativas, para os ramos que movem a construção de variáveis
# para um módulo próprio. O primeiro caminho que existir é o usado.
ALTERNATIVAS: list[list[tuple[str, list[str] | None]]] = [
    [("lai_triagem/variaveis_temporian.py", None),
     ("lai_triagem/codificacao.py", None),
     ("lai_triagem/dados.py", None)],
    [("lai_triagem/variaveis_featuretools.py", None),
     ("lai_triagem/codificacao.py", None),
     ("lai_triagem/dados.py", None)],
]

PADROES = {
    "searchsorted": ["searchsorted"],
    "side=": ["side"],
    "cumsum/cumcount": ["cumsum", "cumcount"],
    "shift": ["shift"],
    "merge_asof": ["merge_asof"],
    "reindex/rótulo": ["reindex", "to_numpy_de_index"],
}


MARCA = "# INVARIANTE:"


def declaradas(fonte: str, faixa: range | None) -> list[str]:
    """Invariantes que o próprio código declara, uma por marca `# INVARIANTE:`.

    Lê os comentários — ao contrário da regra (A), que os descarta —, porque
    aqui o comentário *é* o dado. Cada marca vale uma, mesmo que duas caiam na
    mesma linha lógica: a linha de base também contou por ocorrência, não por
    decisão (os dois `searchsorted` de `organ_rolling` são a mesma ideia e
    contaram dois).
    """
    achadas = []
    for n, linha in enumerate(fonte.splitlines(), 1):
        if faixa is not None and n not in faixa:
            continue
        if MARCA in linha:
            achadas.append(linha.split(MARCA, 1)[1].strip())
    return achadas


def linhas_executaveis(fonte: str) -> dict[int, str]:
    """Devolve {nº da linha: texto}, sem comentário nem docstring.

    Comentário é removido pelo `tokenize`; docstring, pelo `ast`. Sem os dois,
    um comentário que *explica* um `searchsorted` seria contado como um
    `searchsorted`, e a medida inflaria justamente nos trechos bem documentados
    — o incentivo exatamente ao contrário do que se quer.
    """
    arvore = ast.parse(fonte)
    docstrings: set[int] = set()
    for no in ast.walk(arvore):
        corpo = getattr(no, "body", None)
        if isinstance(corpo, list) and corpo and isinstance(corpo[0], ast.Expr) \
                and isinstance(corpo[0].value, ast.Constant) \
                and isinstance(corpo[0].value.value, str):
            alvo = corpo[0]
            docstrings.update(range(alvo.lineno, (alvo.end_lineno or alvo.lineno) + 1))

    fora: dict[int, list[str]] = {}
    for tok in tokenize.generate_tokens(io.StringIO(fonte).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.start[0] in docstrings:
            continue
        fora.setdefault(tok.start[0], []).append(tok.string)
    return {n: " ".join(p) for n, p in fora.items()}


def intervalo_das_funcoes(fonte: str, nomes: list[str] | None) -> range | None:
    """Faixa de linhas ocupada pelas funções pedidas; `None` = arquivo inteiro."""
    if nomes is None:
        return None
    linhas: set[int] = set()
    for no in ast.walk(ast.parse(fonte)):
        if isinstance(no, ast.FunctionDef) and no.name in nomes:
            linhas.update(range(no.lineno, (no.end_lineno or no.lineno) + 1))
    return range(min(linhas), max(linhas) + 1) if linhas else range(0)


def conta(superficie: list[tuple[str, list[str] | None]]) -> tuple[int, list[str]]:
    total = 0
    detalhe: list[str] = []
    for rel, funcoes in superficie:
        caminho = RAIZ / rel
        if not caminho.exists():
            continue
        fonte = caminho.read_text(encoding="utf-8")
        faixa = intervalo_das_funcoes(fonte, funcoes)
        for n, texto in sorted(linhas_executaveis(fonte).items()):
            if faixa is not None and n not in faixa:
                continue
            for rotulo, chaves in PADROES.items():
                for chave in chaves:
                    # `.index.to_numpy()` é preservação de rótulo; um
                    # `to_numpy()` qualquer, não.
                    if chave == "to_numpy_de_index":
                        achados = texto.count("index . to_numpy")
                    else:
                        achados = texto.split().count(chave)
                    for _ in range(achados):
                        total += 1
                        detalhe.append(f"{rel}:{n}  {rotulo}")
    return total, detalhe


def superficie_ativa() -> list[tuple[str, list[str] | None]]:
    for alt in ALTERNATIVAS:
        if (RAIZ / alt[0][0]).exists():
            return alt
    return SUPERFICIE


def conta_declaradas(superficie: list[tuple[str, list[str] | None]]
                     ) -> tuple[int, list[str]]:
    total = 0
    detalhe: list[str] = []
    for rel, funcoes in superficie:
        caminho = RAIZ / rel
        if not caminho.exists():
            continue
        fonte = caminho.read_text(encoding="utf-8")
        for texto in declaradas(fonte, intervalo_das_funcoes(fonte, funcoes)):
            total += 1
            detalhe.append(f"{rel}  {texto}")
    return total, detalhe


if __name__ == "__main__":
    ativa = superficie_ativa()
    por_token, detalhe_token = conta(ativa)
    por_marca, detalhe_marca = conta_declaradas(ativa)
    total = por_token + por_marca

    print("superfície medida:")
    for rel, funcoes in ativa:
        if (RAIZ / rel).exists():
            print(f"  {rel}" + (f"  ({', '.join(funcoes)})" if funcoes else ""))

    print(f"\n(A) pela regra publicada — {por_token}")
    for linha in detalhe_token:
        print(f"  {linha}")
    print(f"\n(B) declaradas no código com `{MARCA}` — {por_marca}")
    for linha in detalhe_marca:
        print(f"  {linha}")

    print(f"\nINVARIANTES MANTIDAS À MÃO: {total}   ({por_token} por token "
          f"+ {por_marca} declaradas)")
    # Argumento opcional: o valor esperado, para uso em guarda.
    if len(sys.argv) > 1 and total != int(sys.argv[1]):
        print(f"FALHOU: esperado {sys.argv[1]}")
        sys.exit(1)
