# Por que um marcador só não basta

> Este documento registra um diagnóstico que **contradiz** a explicação anterior
> (versão 1.0 de `LIMITATIONS.md`), que atribuía o viés de 2 mm à perda de resolução
> dos entalhes interdigitais. Aquela explicação estava errada. O que segue é o que os
> experimentos mostraram.

---

## 1. O sintoma

Com um marcador de 50 × 50 mm e a câmera inclinada 10°, o comprimento medido de um pé
sintético de 265,00 mm saiu **267,07 mm** (+2,07 mm). Com a câmera perpendicular, o
mesmo pé saiu 264,97 mm (−0,03 mm).

## 2. As hipóteses testadas e descartadas

**(a) Resolução do raster retificado.** Se os entalhes entre os dedos não estivessem
sendo resolvidos, aumentar a amostragem resolveria. Não resolveu:

| `rectified_px_per_mm` | erro a 10° de inclinação |
|---|---|
| 6 | +2,07 mm |
| 9 | +2,06 mm |
| 12 | +2,06 mm |

Insensível. A informação não estava sendo perdida na retificação.

**(b) Borramento da reamostragem.** Se o borramento introduzido pelo *warp* deslocasse
a borda, medir a borda na **foto original** (projetando a normal de volta pela
homografia, sem reamostrar) corrigiria. Foi implementado e medido — e **não** corrigiu:
o erro a 10° permaneceu em +2,30 mm, e a mediana do desvio piorou 0,25 mm. O
experimento foi removido do código; sobra o registro aqui.

**(c) Fidelidade da silhueta dos pododáctilos.** A medição por região anatômica
mostrou que o desvio estava concentrado nos dedos, o que sustentava (a) e (b). Mas o
teste decisivo foi outro.

## 3. O teste decisivo

Medir **onde** o contorno se desloca, em vez de quanto:

| Condição | deslocamento nos dedos | deslocamento no calcâneo |
|---|---|---|
| Perpendicular | +0,09 mm | −0,05 mm |
| Inclinada 10° | **−2,20 mm** | −0,12 mm |

O erro estava **inteiro** na extremidade distal. E o marcador, nessa cena, ficava
~350 mm abaixo dos pés — ou seja, perto do calcâneo.

Movendo o marcador para o outro lado da plataforma, perto dos dedos:

| Posição do marcador | deslocamento nos dedos | deslocamento no calcâneo |
|---|---|---|
| Longe dos dedos | −2,20 mm | −0,12 mm |
| Perto dos dedos | +0,09 mm | **+0,54 mm** |

**O erro segue a distância até o marcador.** Não é segmentação, não é borramento, não
é resolução: é **extrapolação da homografia**.

## 4. A explicação

Quatro cantos definem uma homografia de forma exata. "Exata" significa que ela passa
pelos quatro pontos — não que esteja correta em qualquer outro lugar. Qualquer erro
residual na localização subpixel desses cantos (ruído, discretização, refino) produz
uma pequena perturbação angular no plano estimado. Essa perturbação é multiplicada
pela distância: perto do marcador ela é irrelevante; a 350 mm ela vira milímetros.

Pior: com 4 pontos e 8 graus de liberdade, o sistema é exatamente determinado. O
resíduo de reprojeção é **sempre zero**, então a métrica que normalmente indicaria
"esta calibração está ruim" não indica nada. O erro é invisível para o próprio sistema.

É por isso que a verificação de ida-e-volta original — re-medir o lado do marcador na
imagem retificada — dava 0,009 mm enquanto o pé estava 2 mm errado: ela media com
braço de alavanca zero.

## 5. A correção

Distribuir os pontos de controle pela área útil, para que a região dos pés fique
**interpolada** entre marcadores e não extrapolada a partir de um.

O alvo `board4` põe quatro marcadores de 50 mm nos cantos de uma área de 320 × 480 mm.
A homografia passa a ser ajustada por mínimos quadrados sobre **16 pontos**, e:

* o resíduo deixa de ser zero por construção e vira um **sinal real de qualidade**;
* a verificação de ida-e-volta passa a conferir as **distâncias entre marcadores**,
  ao longo de toda a plataforma;
* o sistema sabe dizer se está interpolando ou extrapolando, e quanto.

### Resultado medido

Erro de comprimento, mesma cena geométrica, variando só o alvo de calibração:

| Inclinação | 1 marcador | 4 marcadores |
|---|---|---|
| 0° | −0,03 mm | −0,09 mm |
| 10° | **−1,25 mm** | **−0,07 mm** |
| 18° | **+0,93 mm** | **−0,10 mm** |
| 25° | −0,26 mm | **−0,10 mm** |

Com um marcador o erro é **errático** — muda de sinal com a geometria da captura, o
que impede até uma correção sistemática. Com quatro, é **estável e plano com a
inclinação**, que é a assinatura de um erro residual de segmentação e não de
calibração.

Em cenas de pé único, onde o marcador ficava ainda mais longe, o erro com um marcador
chegava a +2,07 mm; com o tabuleiro, fica em ±0,11 mm.

## 6. O que ficou no sistema

* `calibration/target.py` — a tabela de posições físicas dos marcadores. **É ela que
  define a métrica**; deve ser conferida com paquímetro na plataforma real.
* `calibration/fit.py` — ajuste por mínimos quadrados, resíduo em mm e
  `extrapolation_distance_mm`.
* `quality/gate.py` — verificações `calibration_coverage` e `calibration_residual`.
* `GET /api/marker.pdf?target=board4` — folhas imprimíveis com as coordenadas de
  colagem.
* O modo de marcador único **continua funcionando**, com aviso explícito: quem já tem
  um marcador colado não fica sem sistema, apenas sem a exatidão máxima.

## 7. A mesma geometria, sem impressora

O diagnóstico acima não depende de o ponto de controle ser um ArUco. Qualquer
retângulo de dimensão normalizada — um cartão ISO/IEC 7810, uma folha A4 — fornece a
mesma evidência: quatro cantos com posição física conhecida. E sofre do mesmo
problema: um objeto pequeno ancora a escala num ponto e extrapola sobre os pés; vários
objetos espalhados pela área de apoio devolvem a interpolação.

A diferença é que a **pose** de cada objeto solto é desconhecida, e por isso entra como
incógnita de um ajuste conjunto (`fit.py::fit_free_rectangles`). Detalhes, tolerâncias
das normas e números medidos em **`CALIBRATION_WITHOUT_PRINTING.md`**.

## 8. Lição que vale para o resto do projeto

Uma verificação que só olha para o próprio ponto de referência não verifica nada. A
verificação de escala original era tecnicamente correta e praticamente inútil, porque
tinha braço de alavanca zero. Toda verificação metrológica precisa ser feita **onde a
medida acontece**, não onde a referência está.
