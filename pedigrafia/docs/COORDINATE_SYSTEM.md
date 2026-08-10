# Sistema de coordenadas e transformações

Toda ambiguidade dimensional é eliminada aqui. Qualquer código que produza um número
com unidade física **deve** referenciar um destes frames.

---

## 1. `image` — pixels da foto normalizada

* Origem: canto superior esquerdo.
* `x` → direita, `y` → baixo.
* Obtida da foto original aplicando **somente** a rotação/espelhamento indicado pela
  tag EXIF `Orientation`. Nunca há reamostragem, reescala ou recorte nesta etapa —
  a resolução nativa é preservada.

## 2. `plane` — **milímetros no plano físico da plataforma** (frame canônico)

* Origem: canto **superior esquerdo do marcador**, como devolvido pelo detector.
* `x` → direita ao longo da aresta superior do marcador; `y` → baixo.
* O marcador ocupa exatamente o quadrado `[0, 50] × [0, 50]` mm **por definição**.
* Este é o único frame usado na API, no editor e nas medidas.

Transformação:

```
p_plane_mm = H · p_image_px          (homografia 3×3, coordenadas homogêneas)
H = getPerspectiveTransform(cantos_px, [(0,0), (50,0), (50,50), (0,50)])
```

A ordem dos cantos do ArUco no OpenCV é **TL, TR, BR, BL** no referencial do próprio
marcador, o que torna a correspondência acima direta e independente da rotação da
câmera.

> Coordenadas `plane` podem ser negativas (a plataforma se estende para todos os lados
> do marcador). Isso é esperado e correto.

## 3. `rect` — pixels do raster retificado

Existe **apenas** para rasterização (segmentação, exibição, PNG anotado).

```
p_rect_px = (p_plane_mm - origin_mm) * px_per_mm
p_plane_mm = origin_mm + p_rect_px / px_per_mm
```

* `px_per_mm` = `settings.rectified_px_per_mm` (padrão **6,0 px/mm** ≈ 152,4 dpi).
* `origin_mm` = canto superior esquerdo da janela de trabalho retificada, escolhida
  como a interseção entre (a) a projeção da imagem no plano e (b) a área de trabalho
  física máxima `working_area_mm` centrada no conteúdo, e depois **quantizada** para
  1/`px_per_mm` mm para que a conversão seja exata em ambos os sentidos.
* A transformação é **afim com escala uniforme**: não distorce, não altera proporções,
  e é perfeitamente invertível. Todo erro introduzido aqui é de *amostragem*
  (≤ 1/(2·px_per_mm) = 0,083 mm), não de escala.

## 4. `foot` — frame anatômico local (mm)

Definido por pé, após a detecção do eixo longitudinal:

* `origin` = ponto do contorno com menor projeção em `u` (extremo posterior do calcâneo),
  projetado sobre o eixo.
* `u` = vetor unitário do eixo longitudinal, apontando **do calcâneo para os dedos**.
* `v` = `rot90(u)` = `(-u.y, u.x)`.
* Coordenadas locais: `u_p = (p - origin)·u`, `v_p = (p - origin)·v`.
* `t = u_p / lengthMm ∈ [0, 1]` — posição longitudinal normalizada.

As posições das cabeças metatarsais M1–M5 são reportadas **neste frame**
(`landmarksFootFrame`), conforme exigido, além do frame `plane`.

O sinal de `v` para medial/lateral depende da lateralidade e do ponto de vista
(`view = below` para podoscópio). O campo `medialSign ∈ {+1, -1}` no `FootFrame`
diz de que lado está a borda medial, e é o único lugar do código que conhece essa
convenção.

## 5. `page` — folha A4 do PDF

```
p_page_mm = R(θ) · (p_plane_mm - c_plane) + c_page        (rígida: rotação + translação)
p_pt      = p_page_mm × 72 / 25.4
```

* `R(θ)` é uma rotação pura (det = +1). Nenhum fator de escala é aplicado em nenhum
  momento — `det(R) = 1` é verificado em teste.
* A página tem MediaBox `[0, 0, 595.2755905511812, 841.8897637795277]` pt
  = 210,000 × 297,000 mm.
* O eixo `y` do PDF aponta **para cima**; a conversão inverte `y` uma única vez, no
  `pdf/builder.py`, e essa inversão também é rígida (reflexão + rotação → mantida como
  isometria; distâncias preservadas).

## 6. Tabela de conversões (constantes exatas)

| De → Para | Fator |
|---|---|
| mm → pt | `× 72 / 25.4` = `× 2.834645669291339` |
| pt → mm | `× 25.4 / 72` = `× 0.35277777777777775` |
| mm → px (rect) | `× px_per_mm` (padrão 6,0) |
| polegada | `25.4 mm` (exato, por definição) |

## 7. Regras invioláveis

1. Nenhuma função pode devolver "tamanho" sem que a unidade esteja no nome
   (`*_mm`, `*_px`, `*_pt`, `*_mm2`).
2. Nenhum código de UI, CSS, canvas ou PDF pode aplicar escala à geometria física.
   O zoom do editor é uma matriz de **view**, aplicada no momento do desenho e nunca
   persistida no modelo.
3. Conversões px↔mm só existem em `calibration/homography.py` (`Rectification`) e em
   `frontend/src/geom/units.ts`. Qualquer outro `/ px_per_mm` no código é um bug.
