# Calibrar sem imprimir: objetos de dimensão normalizada

O alvo ArUco de quatro marcadores continua sendo o método mais exato do sistema.
Este documento descreve o caminho alternativo — usar um objeto que já existe, de
dimensão padronizada — e diz **exatamente** quanto se perde em cada opção.

> A regra não muda: a escala vem de uma referência física de dimensão conhecida.
> Uma fotografia sozinha não contém tamanho. O que muda é de onde vem essa
> referência.

---

## 1. O que entra e o que não entra

### Entra: retângulos planos de dimensão normalizada

Quatro cantos com posição física conhecida são exatamente o que a homografia
precisa — é a mesma evidência que um marcador ArUco fornece.

| Objeto | Dimensão | Norma | Tolerância da norma | Incerteza de escala | Em um pé de 265 mm |
|---|---|---|---|---|---|
| Cartão (crédito, débito, RG novo) | 85,60 × 53,98 mm | ISO/IEC 7810 ID-1 | ±0,13 mm | 0,24 % | **±0,64 mm** |
| Folha A4 | 210 × 297 mm | ISO 216 | ±2 mm | 0,95 % | **±2,52 mm** |
| Folha A5 | 148 × 210 mm | ISO 216 | ±2 mm | 1,35 % | **±3,58 mm** |
| Retângulo cortado sob medida | declarada | — | do seu paquímetro | — | — |

Essa coluna final é um **piso de erro**: é a incerteza do próprio padrão físico,
e nenhum processamento de imagem a remove. Um cartão é um padrão de comprimento
surpreendentemente bom; uma folha de papel, não — o corte industrial de papel
simplesmente não é preciso.

O sistema reporta esse número em `calibration.scaleToleranceMm` e o exibe no painel
de revisão.

### Não entra: moeda

Foi pedida e foi recusada, por aritmética.

A projeção de um círculo é uma elipse. Sem os parâmetros intrínsecos da câmera, a
elipse dá a escala apenas **assumindo que o plano está frontal**. O erro dessa
suposição, para um ponto a distância `d` da moeda, com o plano inclinado de `θ` e a
câmera a distância `Z`, é

```
erro relativo ≈ d · sen θ / Z
```

Com `d = 150 mm` (moeda ao lado do pé), `θ = 3°` e `Z = 600 mm`: **1,3 %** — ou seja,
**3,4 mm** em um pé de 265 mm.

E 3° é indetectável pela própria moeda: a razão dos eixos da elipse seria
`cos 3° = 0,9986`, uma diferença de 0,14 % que se perde no ruído de ajuste da elipse.

Ou seja: a moeda não erra pouco — ela erra **sem avisar**. Um sistema que não pode
medir o próprio erro não deve reportar o resultado como medida.

### Não entra: régua

Os quatro cantos de uma régua são quase colineares em uma das direções. A homografia
fica malcondicionada exatamente na direção em que precisaria corrigir a perspectiva.
Ler as marcações da régua resolveria o problema geométrico, mas troca o problema por
um OCR de traços finos em foto de celular — que é menos confiável, não mais.

---

## 2. O problema difícil não é achar o retângulo — é saber qual ele é

O detector localiza quadriláteros e refina os cantos por ajuste de retas nas arestas
(veja §4). Isso funciona bem. O problema é outro: **um retângulo não diz quanto mede**.

A razão largura/altura do retângulo original é recuperável da perspectiva — é o
resultado de Zhang & He (2007), implementado em
`calibration/reference.py::rectangle_aspect`:

* sejam `g1, g2` as duas primeiras colunas da homografia do quadrado unitário para o
  quadrilátero observado, com origem no ponto principal;
* se o objeto é um retângulo, `K⁻¹g1 ⊥ K⁻¹g2`, o que fixa a distância focal:
  `f² = −(g1ₓg2ₓ + g1ᵧg2ᵧ) / (g1_z g2_z)`;
* e daí `L/A = ‖K⁻¹g1‖ / ‖K⁻¹g2‖`.

Medido em cenas sintéticas, isso recupera a razão com erro **< 1 %** e a distância
focal com erro < 5 %. Bom — mas insuficiente para identificar o objeto, por dois
motivos:

**(a) Degenerescência geométrica.** Quando o eixo de inclinação da câmera coincide com
um lado do retângulo, aquele par de lados continua paralelo na imagem: o ponto de fuga
vai ao infinito e `f` deixa de ser observável. O erro da razão sobe para 4–9 %.

O módulo **mede** esse malcondicionamento — propaga ruído de 1 px pelos cantos e
reporta o desvio-padrão da razão — em vez de devolver um número com cara de medida.

**(b) A série ISO A inteira tem a mesma razão.** A4, A5, A3 — todas √2. Nenhuma
fotografia distingue uma da outra pela forma. E um cartão fica a apenas ~11 % dessa
razão, o que é da ordem do erro em (a).

Confundir cartão com folha A4 erra a escala em **2,45×**. O molde sairia coerente,
bonito e completamente errado.

### O que separa o retângulo certo dos quadriláteros falsos

Dois filtros, ambos diretos:

1. **Resíduo do ajuste das arestas.** As quatro arestas que o candidato afirma ter
   precisam existir. Medido em cena sintética: o cartão correto ajusta suas retas com
   resíduo de **0,15 px**; recortes espúrios do `approxPolyDP` ficam em **6,7–8,7 px**.
   O corte em 2,0 px separa os dois grupos com folga.
2. **Razão de aspecto**, contra a dimensão declarada, com a tolerância vinda da
   própria incerteza da medida.

Um terceiro filtro — plausibilidade da distância focal recuperada — foi implementado
e depois **removido por medição**. Quanto mais perpendicular a fotografia (que é
exatamente o que o sistema pede ao operador), mais `f` tende ao infinito, e o limite
superior passava a rejeitar o candidato certo: numa cena de 1,6° de inclinação o
cartão correto saía com `f/máx = 6,5` e era descartado, enquanto um quadrilátero
espúrio sobrevivia por acaso. Um teste que falha justamente na condição recomendada
de captura não é um teste.

Por isso: **o modo automático se recusa a escolher** quando a forma não prova de qual
objeto se trata, e a interface pede que o operador declare. Não é burocracia — é a
única barreira real contra a falha silenciosa mais perigosa deste caminho.

### As redes de segurança — e o buraco que fica

Não existe verificação interna capaz de detectar uma referência trocada: a medida sai
internamente coerente. Sobram duas barreiras indiretas:

1. **O filtro de razão de aspecto**, quando as formas diferem o bastante. Cartão
   (1,586) contra A4 (1,414) diferem 12 %, e o quadrilátero é recusado.
2. **O limite físico de comprimento plantar** (90–400 mm, `GeometryThresholds`):
   declarar "A4" para uma foto de cartão faria os pés medirem ~650 mm, e a captura é
   reprovada em vez de produzir um molde errado.

**O buraco que nenhuma das duas fecha:** toda a série ISO A tem proporção √2. Declarar
A4 para uma folha **A5** multiplica a escala por 210/148 = **1,42**; um pé de 265 mm
sairia com 376 mm — abaixo do limite de 400 mm, portanto **sem nenhum alarme**.

O sistema não pode detectar isso. O que ele faz é dizer, em aviso explícito na análise,
que o formato da folha não é verificável na foto e precisa de conferência humana.
Está travado em teste (`test_iso_a_series_hazard_is_declared`).

É mais um motivo para preferir o cartão: a razão 85,60/53,98 não é compartilhada com
nenhum outro objeto de bolso, e a tolerância dele é 15× menor que a do papel.

---

## 3. Um objeto pequeno traz de volta a extrapolação

É o mesmo problema diagnosticado em `METROLOGY_CALIBRATION.md`. Um cartão é pequeno e
fica ao lado dos pés: a homografia é exata sobre ele e **extrapola** sobre a região que
importa.

O sistema resolve isso do mesmo jeito, sem impressora: **vários objetos iguais**,
espalhados ao redor da área de apoio.

Com dois ou mais, o ajuste passa a ser conjunto (`fit.py::fit_free_rectangles`):

* incógnitas: a homografia (8) + a pose no plano de cada objeto extra (3 cada);
* resíduos: 8 por objeto (os quatro cantos contra a forma declarada);
* referencial: o do primeiro objeto, o que fixa a liberdade global de similaridade.

Com dois objetos são 11 incógnitas contra 16 resíduos — sobra informação, e o resíduo
deixa de ser zero por construção e passa a ser um sinal real de qualidade. É por isso
que `calibration.exact` vira `false` e a verificação `calibration_residual` entra em
cena.

**Geometria importa mais que quantidade.** Dois cartões pequenos cobrem uma *faixa*;
o casco convexo dos pontos de controle é quase uma reta, e os pés continuam a alguma
distância dele. Quatro, nos cantos da área de apoio, cobrem uma *área* — e aí os pés
ficam interpolados, exatamente como no alvo impresso.

---

## 4. Cantos que não existem

Um cartão ID-1 tem cantos arredondados com raio de **3,18 mm**. O vértice ideal do
retângulo **não existe** na imagem — não há pixel para o `cornerSubPix` encontrar.

A solução é não procurar o canto:

1. amostrar perfis perpendiculares ao longo de cada aresta, descartando as pontas (ao
   menos o dobro do raio de arredondamento);
2. localizar a borda em cada perfil pelo máximo do gradiente, com interpolação
   parabólica (subpixel);
3. ajustar a reta por mínimos quadrados totais, com uma passada de rejeição de
   *outliers*;
4. **intersectar** as quatro retas.

O vértice sai da geometria, não da imagem. Medido nas cenas sintéticas: resíduo RMS do
ajuste das arestas **< 0,2 px**.

O gerador de cenas renderiza o arredondamento de verdade (`_rounded_rect_mm`), para que
esse requisito fique visível no teste em vez de virar suposição.

---

## 5. Exatidão medida

Metodologia idêntica à de `METROLOGY_CALIBRATION.md`: cenas sintéticas de geometria
conhecida, inclinação de 0 a 24°, rolagem ±15°, distância de 520 a 1050 mm, ruído
σ = 1–5. Erro de comprimento = maior corda do casco convexo (medida estacionária).

<!-- RESULTADOS -->

---

## 6. Como usar na prática

**Sem impressora, o melhor arranjo:** quatro cartões iguais, um em cada canto da área
de apoio, no mesmo plano da planta (sobre o vidro do podoscópio, ao lado dos pés).
Declare "Cartão" na captura.

**Com um cartão só:** funciona, e o sistema informa a distância em que está
extrapolando. Serve para conferência; para fabricação, prefira o alvo impresso ou
mais objetos.

**Folha A4:** cobre bem a área, mas carrega ±2 mm de tolerância de corte. Se for usar,
**meça a folha com paquímetro** e informe as dimensões reais
(`PEDIGRAFIA_REFERENCE_CUSTOM_MM="209.4x296.7@0.2"`) — isso elimina o termo dominante
de erro dessa opção.

**Regra geral:** o objeto precisa estar plano, inteiro, no mesmo plano físico das
plantas, e nunca sobreposto aos pés.

---

## 7. Configuração

| Variável | Padrão | Efeito |
|---|---|---|
| `PEDIGRAFIA_REFERENCE_OBJECT` | `auto` | `auto`, `off`, `card`, `a4`, `a5` ou `"LxA"` |
| `PEDIGRAFIA_REFERENCE_CUSTOM_MM` | — | `"85.6x53.98@0.1"` (largura × altura @ tolerância) |
| `PEDIGRAFIA_PRINTED_TARGET_TOLERANCE_MM` | `0.20` | Tolerância assumida do alvo impresso |

Na API: `POST /api/analyze` aceita `calibration=aruco|card|a4|a5|auto`.
`GET /api/references` publica o catálogo, as tolerâncias e os motivos das recusas.
