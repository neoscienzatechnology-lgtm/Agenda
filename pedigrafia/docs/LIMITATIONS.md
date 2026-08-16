# Limitações atuais e exatidão medida

> **Declaração obrigatória:** este sistema **não** teve precisão clínica nem
> metrológica validada. Todos os números abaixo vêm de **cenas sintéticas** com
> geometria conhecida por construção. Nenhum ensaio foi feito com hardware real,
> pés reais ou impressões físicas. Antes de qualquer uso clínico ou de fabricação,
> execute o `VALIDATION_CHECKLIST.md` com régua e marcador impresso.

---

## 1. Exatidão medida em cenas sintéticas

### 1.1 Cadeia metrológica (marcador → milímetro)

Verificada re-detectando o marcador na imagem retificada e medindo seus lados:

| Inclinação da câmera | Erro máximo do lado de 50,00 mm |
|---|---|
| 0° | 0,005 mm |
| 12° | 0,032 mm |
| 22° | 0,028 mm |
| 34° | 0,072 mm |

**A conversão pixel → milímetro não é a fonte de erro do sistema.**

> **Correção importante (v1.1).** A versão anterior deste documento atribuía o viés
> com câmera inclinada à perda de resolução dos entalhes interdigitais. **Aquela
> explicação estava errada.** A causa medida é a **extrapolação da homografia** a
> partir de um único marcador — ver `METROLOGY_CALIBRATION.md`. Com o alvo de quatro
> marcadores o erro cai para ±0,11 mm e deixa de depender da inclinação.

### 1.2 Comprimento do pé, fim a fim (foto sintética → mm)

| Condição | Erro de comprimento |
|---|---|
| Perpendicular, 200/240/260/265/270 mm | −0,04 a +0,12 mm |
| Distância 480 mm ↔ 1100 mm | dispersão < 0,2 mm |
| Baixa resolução (1200 × 1560) | +0,10 mm |
| JPEG qualidade 70 | −0,03 mm |
| Ruído σ = 6 | +0,12 mm |
| Fundo escuro (podoscópio retroiluminado) | −0,01 mm |
| Rolagem de câmera 8° | +1,05 mm ¹ |
| Rolagem de câmera 17° | +1,67 mm ¹ |
| Inclinação 10° | +2,07 mm ¹ |
| Inclinação 20° | +0,30 mm ¹ |

¹ Com **marcador único**. Com o alvo de quatro marcadores (`board4`), as mesmas
condições dão −0,07 a −0,11 mm, sem dependência da inclinação:

| Inclinação | 1 marcador | 4 marcadores |
|---|---|---|
| 0° | −0,03 mm | −0,09 mm |
| 10° | −1,25 mm | −0,07 mm |
| 18° | +0,93 mm | −0,10 mm |
| 25° | −0,26 mm | −0,10 mm |

### 1.3 PDF (geometria em mm → arquivo)

| Comprimento | Erro relido do arquivo |
|---|---|
| 200 / 240 / 260 / 265 / 270 mm | 0,0000 mm |

Invariância de distâncias par-a-par no posicionamento na página: **1,1 × 10⁻¹³ mm**.
MediaBox: 595,2756 × 841,8898 pt = 210,000 × 297,000 mm.

---

## 2. A limitação dominante: cobertura da calibração

O erro do sistema **não** está na conversão pixel→milímetro nem na segmentação. Está
em **onde os pontos de controle estão** em relação à região medida.

Com um marcador só, a homografia é exata sobre ele e extrapola para o resto da
plataforma; o erro cresce com a distância e muda de sinal conforme a geometria da
captura. Com quatro marcadores ao redor da área de apoio, os pés ficam interpolados e
o erro fica estável em ±0,11 mm.

O diagnóstico completo, com os experimentos que descartaram as hipóteses de resolução
e de borramento, está em **`METROLOGY_CALIBRATION.md`**.

**Consequências práticas**

* Use o alvo de quatro marcadores (`GET /api/marker.pdf?target=board4`) para qualquer
  medida destinada a fabricação.
* O modo de marcador único continua disponível e avisa explicitamente que está
  extrapolando, informando a distância em milímetros.
* As **posições físicas** dos marcadores na plataforma são o que calibra o sistema.
  Meça-as com paquímetro e ajuste o arquivo do alvo — não confie no desenho.

### 2.1 Erro residual após a correção

Com o tabuleiro, o que resta (±0,11 mm, sistematicamente negativo) é erro de
segmentação: a fronteira do contorno cai fração de milímetro para dentro. É uma ordem
de grandeza menor que a variabilidade de posicionamento do pé entre capturas, e por
isso não é hoje o gargalo de exatidão.

## 3. Outras limitações conhecidas

### 3.1 Estimativa de inclinação é aproximada
`marker.tiltDeg` é calculado assumindo distância focal `f ≈ 1,15 × maior dimensão`,
pois não há calibração intrínseca da câmera. Em câmeras teleobjetivas o valor
**subestima** a inclinação real (medido: 22° reportados para 34° reais). É um sinal
de qualidade, nunca uma medida.

### 3.2 Reflexo intenso pode recortar a planta
Com brilho especular forte sobre o vidro (testado com refletância 0,45), a área
plantar detectada caiu ~10 % porque a mancha especular tocou a borda do pé e foi
excluída como fundo. O comprimento permaneceu correto (−0,02 mm), mas a área e o
índice de arco ficaram subestimados. O *quality gate* sinaliza reflexo; a revisão
manual permite corrigir o contorno.

### 3.3 M2, M3 e M4 são estimados por modelo
Cabeças metatarsais não são visíveis em fotografia plantar. M1 e M5 vêm de extremos
reais do contorno (proeminências do antepé); M2–M4 são interpolados por um modelo
paramétrico explícito (protuberância distal de 3,0 % do comprimento, ápice em
s ≈ 0,30). Saem com `method="interpolated_model"` e confiança ≤ 0,5, e a UI os
destaca. **Nunca** devem ser lidos como observação.

### 3.4 Índice de arco é de silhueta, não de contato
A razão de áreas de Cavanagh calculada sobre a **silhueta** dá valores
sistematicamente mais altos (~0,32 no modelo sintético) do que a literatura de
pedigrafia de contato (~0,21–0,26), porque a silhueta inclui a região do arco que não
toca o solo. Está rotulado como `archIndexBasis: "silhouette"` e acompanhado de aviso.
Um índice de contato exigiria segmentação validada da área de isquemia — implementada
como `detect_contact_region` mas **não** usada como base por padrão (confiança baixa).

### 3.5 Linha de corte dos pododáctilos frequentemente cai no padrão
A detecção automática exige um vale real no perfil de largura com rebote. Em
silhuetas onde os dedos são contíguos ao antepé (o caso comum) não há vale, e o
sistema usa a proporção padrão t = 0,78 com confiança 0,20, sinalizando revisão. Isso
afeta apenas o índice de arco e as faixas das zonas de apoio — nenhuma medida linear.

### 3.6 Segmentação é clássica, não treinada
O segmentador padrão é OpenCV (distância cromática ao fundo + Otsu + GrabCut como
filtro). Ele foi validado contra cenas sintéticas com fundo claro e escuro, mas
**não** contra fotografias reais de podoscópio, com meias, sombras, tatuagens,
esmalte, pelos, variação de tom de pele ou patologias. Ver `docs/TRAINING.md`.

### 3.7 Escopo de processo
* Sessão em memória de **processo único** — não escala horizontalmente sem
  substituir `storage.py` por um store compartilhado.
* Rate limiting em memória, por instância.
* Sem login, banco, histórico ou multiusuário (fora do escopo do MVP, por decisão).
* HEIC/HEIF só decodifica se o Pillow do ambiente tiver o plugin; caso contrário o
  upload é recusado com mensagem clara.

### 3.8 Espelhamento do molde
Para captura de podoscópio (`view="below"`) o PDF é **espelhado** de propósito, para
que a folha impressa possa ser usada com a face para cima como gabarito físico. Se o
seu fluxo de fabricação exigir a orientação oposta, use `view="above"`. Isso está
verificado em teste (`test_view_controls_mirroring`), mas **confirme com uma peça
real antes de produzir em série**.
