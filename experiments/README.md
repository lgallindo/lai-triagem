# Experimento: modelo em dois estágios

**Branch `experimento/dois-estagios`. Material para publicação futura.**
`main` segue com estágio único, e o resultado abaixo explica por quê.

Executar: `uv run python experiments/dois_estagios.py` (~11 s).

## Hipótese

45,8% das linhas não têm histórico aproveitável do solicitante. No estágio único
essas linhas recebem o sentinela `-1` em oito variáveis, e um só conjunto de
divisões tem de servir aos dois regimes. Um modelo por regime deveria usar
melhor a capacidade do que um modelo forçado a acomodar ambos.

## A pergunta que importa não é se cada modelo fica melhor

O entregável é **uma** fila ranqueada — "os 5% mais arriscados". Dois modelos
treinados em subpopulações diferentes produzem escores em **escalas
incomparáveis**, e não existe "top 5% conjunto" de escores fora de escala comum.
Por isso o experimento mede três coisas.

## Resultados (teste 2026 maturado, n=86.096, taxa-base 5,75%)

Cobertura do regime com histórico: **47,9%** do teste. Taxa de reencaminhamento
**5,37%** com histórico contra **6,10%** sem.

### E1 — cada modelo no seu próprio regime

| Modelo | n | PR-AUC | AUC | p@1% | p@5% |
|---|---|---|---|---|---|
| A, com histórico (31 var) | 41.219 | **0,3527** | 0,8499 | 0,6942 | 0,3862 |
| estágio único, no mesmo subconjunto | 41.219 | 0,3470 | 0,8537 | 0,6845 | 0,3862 |
| B, sem histórico (23 var) | 44.877 | 0,1828 | 0,7407 | 0,3474 | 0,2473 |
| estágio único, no mesmo subconjunto | 44.877 | 0,1831 | 0,7452 | 0,3586 | 0,2469 |

**A hipótese se confirma, mas fracamente:** o modelo específico ganha **+0,0057**
de PR-AUC no seu regime. O modelo B não ganha nada — empata com o estágio único,
o que faz sentido, porque no regime sem histórico as duas configurações têm
acesso à mesma informação.

### E2 — fila única com escores crus: incomparabilidade medida

| Regime | mín | p50 | p95 | máx |
|---|---|---|---|---|
| A, com histórico | 0,0080 | 0,0296 | 0,2395 | 0,8186 |
| B, sem histórico | 0,0231 | 0,0693 | 0,1749 | 0,6066 |

As distribuições não coincidem: a mediana de B é **2,3× a de A**, mas a cauda de
A vai bem mais longe. O efeito é direto e quantificável:

> **O top 5% agrupado fica com 66,6% de linhas do regime A, quando o regime A é
> 47,9% da população.**

A fila crua está enviesada em quase 19 pontos percentuais por artefato de
escala, não por risco. Isso é um problema de equidade, não só de métrica: o
regime B concentra solicitantes anonimizados e de primeira viagem, exatamente o
grupo que o Termo de Abertura quer proteger.

### E3 — fila única com escores calibrados por regime

Calibração isotônica ajustada na validação, uma por regime.

> **O top 5% calibrado fica com 51,6% do regime A** — contra 47,9% da população.

A calibração **conserta a coerência da fila**: o desvio cai de 18,7 pp para
3,7 pp. Isto é o achado de desenho mais importante do experimento.

### Veredito

| Desenho | PR-AUC | p@1% | p@5% | p@10% |
|---|---|---|---|---|
| Estágio único (`main`) | 0,2619 | 0,5424 | 0,3136 | **0,2362** |
| Dois estágios, crus | **0,2639** | 0,5587 | 0,3124 | 0,2358 |
| Dois estágios, calibrados | 0,2532 | **0,5633** | **0,3148** | 0,2357 |

O melhor por precisão@5% é o duplo calibrado, com **+0,0012** — doze centésimos
de ponto percentual. Está dentro do ruído, e vem acompanhado de PR-AUC **pior**
(0,2532 contra 0,2619).

**Conclusão: o ganho intrarregime não sobrevive ao agrupamento.** A necessidade
de produzir uma fila única sob capacidade fixa dissolve a vantagem que cada
modelo tinha no seu domínio.

## Por que `main` fica com estágio único

Custos que o experimento não mede e que pesam contra:

- duas etiquetas BentoML, dois ciclos de vida, dois limiares a calibrar;
- a tabela de calibração migra de enfeite de leitura para **parte crítica do
  caminho de ranqueamento** — se ela envelhecer, a fila enviesa, e o
  experimento mostra que o viés chega a 19 pp;
- dois modelos a reauditar (vazamento, equidade) a cada retreinamento.

Trocar 0,12 pp de precisão@5% por isso não se justifica.

## O que aqui é publicável

Três coisas que não encontrei na literatura varrida
([`../docs/LITERATURE.md`](../docs/LITERATURE.md)):

1. **Modelagem por regime de disponibilidade melhora o regime e não melhora o
   produto**, quando o produto é uma fila limitada por capacidade. Resultado
   negativo com mecanismo identificado, não apenas ausência de efeito.
2. **Incomparabilidade de escala é um problema de equidade, não de métrica.**
   O desvio de 18,7 pp na composição da fila recai sobre o grupo mais
   vulnerável — anonimizados e estreantes.
3. **Calibração como instrumento de coerência de fila, não de acurácia.** Em
   `main` a calibração custa 0,56 pp de precisão@5% e não compra nada; aqui ela
   não compra precisão tampouco, mas corrige 15 pp de viés de composição. É o
   mesmo instrumento com duas finalidades distintas, e a literatura de
   calibração costuma discutir só a primeira.

Para retomar: `git checkout experimento/dois-estagios`.
