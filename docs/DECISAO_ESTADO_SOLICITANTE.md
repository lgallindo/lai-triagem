# Decisão de arquitetura: onde vive o histórico do solicitante

**Situação:** decidida. **Data:** 16/09/2026. **Revisar se:** a CGU autorizar
retenção de perfil por requerente no artefato, ou se a cobertura de
`IdSolicitante` mudar materialmente.

## Contexto

A engenharia de variáveis (ver [`VERIFICATION.md`](VERIFICATION.md)) mostrou que
o histórico do solicitante é o maior ganho legítimo do projeto: **+0,0434 de
PR-AUC** no teste maturado, e a experiência **específica do órgão**
(`n_pedidos_previos_neste_orgao`) vale 6,67% do ganho contra 1,29% do agregado.
O mecanismo é claro: quem já usou a LAI aprende qual órgão endereçar —
estreantes são reencaminhados a 8,33%, veteranos com 50+ pedidos a 5,35%.

Usar essas variáveis exige que o escore conheça, no instante da chegada, quantos
pedidos aquele cidadão já fez e quantos foram devolvidos.

## Correção de um erro de registro

Versões anteriores deste repositório afirmavam que o serviço era
"deliberadamente sem estado". **Isso era impreciso e a afirmação foi promovida
indevidamente de conveniência a princípio.** O histórico real: na v1 descartei o
contador por solicitante por simplicidade, sob a restrição declarada de que
velocidade de treinamento tinha precedência, e passei a repetir a ausência como
se fosse decisão fundamentada.

Dois enganos embutidos nessa formulação:

1. **"Com estado" versus "sem estado" é falso binário.** O histórico pode ser uma
   tabela somente-leitura reajustada periodicamente, exatamente como
   `orgao_rate`. A diferença entre os dois casos não é de natureza, é de
   cardinalidade.
2. **Não havia obstáculo técnico.** Medido: apenas **51.319** solicitantes têm
   histórico aproveitável (≥2 pedidos), porque 78,7% dos 240.407 pediram uma
   única vez. A tabela correspondente ocupa **~0,4 MiB em parquet** — duas ordens
   de grandeza abaixo do que "precisa de estado" sugeria.

| | `orgao_rate` | histórico do solicitante |
|---|---|---|
| Chaves | ~880 órgãos | 51.319 cidadãos |
| Natureza da chave | entidade pública | **pessoa natural identificada** |
| Reajuste | periódico, tabela | periódico, tabela |
| Custo | 178 KB | ~0,4 MiB |

Removido o argumento técnico, resta apenas uma objeção — e ela é substantiva.

## A objeção que de fato importa

Embarcar a tabela significa o artefato carregar, por cidadão identificado,
**quantas vezes ele pediu informação ao Estado e quantas vezes foi mal
roteado**. Num sistema cuja finalidade é o cidadão fiscalizar o Estado, isso
inverte a direção do escrutínio: o Estado passa a manter, dentro de um escore
operacional, um registro de comportamento de quem o fiscaliza.

É problema de proteção de dados e de desenho institucional, não de engenharia.
Incide a LGPD, e a finalidade declarada do tratamento — priorizar fila de
triagem — não autoriza por si a construção de perfil histórico por requerente
num artefato que circula fora do banco de origem.

Atenuante relevante: **o Fala.BR já possui esses dados.** O contador seria
derivado do banco da própria CGU. A pergunta correta não é "criar retenção
nova?", mas "qual uso é legítimo da retenção existente?".

## Decisão

Separar as variáveis por **natureza do titular do dado**, não por conveniência
de implementação.

**1. Lado do órgão — embarcado no artefato.** `orgao_rate`,
`orgao_rate_movel_90d`, `orgao_rate_movel_365d`,
`dias_desde_primeiro_pedido_do_orgao`. Todas agregam conduta de **entidades
públicas**, sobre as quais não há expectativa de privacidade. Ficam em
`artifacts/preprocessor.json` e são reajustadas a cada retreinamento.

**2. Lado do solicitante — fornecido pelo chamador, nunca retido.** As seis
variáveis de histórico passam a ser **campos opcionais de entrada** da API. O
Fala.BR, que já detém o dado, informa os contadores na chamada. O artefato **não
contém nenhum dado pessoal** e o serviço não persiste nada.

**3. Degradação explícita.** Ausente o histórico, os campos valem `-1`, que o
LightGBM trata como faltante — o mesmo valor usado no treinamento para
solicitante anonimizado. O modelo não quebra; perde precisão de forma
mensurável.

## Consequências

**Favoráveis.** O artefato permanece livre de dado pessoal e continua
distribuível por git puro. A retenção fica onde já estava, no banco da CGU, sob
o controle de acesso existente. O ganho de precisão é capturado quando o
chamador opta por informar os contadores, e a opção é dele, não nossa. Auditoria
fica mais simples: não há perfil a auditar dentro do modelo.

**Desfavoráveis.** O chamador precisa calcular seis contadores, o que transfere
complexidade para a integração. Um chamador que não os informe recebe um modelo
de desempenho reduzido, sem aviso além da resposta. E a cobertura limita o ganho
de todo modo: **45,8% das linhas não têm histórico aproveitável** (16,9%
anonimizadas, 28,9% de quem pediu uma vez só), então o efeito medido vem de
pouco mais da metade do volume.

**Explicitamente rejeitado.** Embarcar a tabela de 51.319 cidadãos no artefato —
tecnicamente trivial (0,4 MiB), institucionalmente indefensável sem autorização
expressa da CGU. Se essa autorização vier, a decisão se revisa e a mudança é de
uma linha no exportador.

## Como verificar que a decisão está implementada

```bash
# O artefato não deve conter nenhuma chave de solicitante:
uv run python -c "
import json; m=json.load(open('artifacts/preprocessor.json'))
assert 'solicitante_hist' not in m, 'dado pessoal embarcado!'
print('tabelas embarcadas:', [k for k in m if 'rate' in k or 'birth' in k])
print('campos de histórico esperados do chamador:', m['caller_supplied_features'])
"
```

O `/health` do serviço também expõe `caller_supplied_features`, de modo que um
integrador descobre o contrato sem ler o código.
