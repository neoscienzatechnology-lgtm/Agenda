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

### 1.2 Comprimento do pé, fim a fim (foto sintética → mm)

| Condição | Erro de comprimento |
|---|---|
| Perpendicular, 200/240/260/265/270 mm | −0,04 a +0,12 mm |
| Distância 480 mm ↔ 1100 mm | dispersão < 0,2 mm |
| Baixa resolução (1200 × 1560) | +0,10 mm |
| JPEG qualidade 70 | −0,03 mm |
| Ruído σ = 6 | +0,12 mm |
| Fundo escuro (podoscópio retroiluminado) | −0,01 mm |
| Rolagem de câmera 8° | **+1,05 mm** |
| Rolagem de câmera 17° | **+1,67 mm** |
| Inclinação 10° | **+2,07 mm** |
| Inclinação 20° | +0,30 mm |

### 1.3 PDF (geometria em mm → arquivo)

| Comprimento | Erro relido do arquivo |
|---|---|
| 200 / 240 / 260 / 265 / 270 mm | 0,0000 mm |

Invariância de distâncias par-a-par no posicionamento na página: **1,1 × 10⁻¹³ mm**.
MediaBox: 595,2756 × 841,8898 pt = 210,000 × 297,000 mm.

---

## 2. A limitação dominante: fidelidade da silhueta dos pododáctilos

O erro do sistema **não** está na calibração. Está em quão fielmente a silhueta dos
dedos sobrevive à captura.

**Mecanismo.** Os entalhes interdigitais têm poucos milímetros de largura. Quando o
borramento efetivo da foto cresce (inclinação ou rolagem da câmera, profundidade de
campo, reamostragem), esses entalhes deixam de ser resolvidos e a silhueta "preenche"
o vão entre os dedos. O contorno resultante fica **inflado para fora**, o que aumenta
o comprimento medido.

Medido por região anatômica (inclinação 10°, desvio mediano em relação à verdade):

| Região | Perpendicular | Inclinada 10° |
|---|---|---|
| Calcâneo | 0,141 mm | 0,204 mm |
| Mediopé | 0,199 mm | 0,237 mm |
| Antepé | 0,147 mm | 0,213 mm |
| **Pododáctilos** | **0,144 mm** (p95 1,75) | **1,104 mm** (p95 3,63) |

**Aumentar a resolução da retificação não corrige** (testado a 6, 9 e 12 px/mm: erro
idêntico em 0,01 mm). A informação já se perdeu na fotografia.

**Consequências práticas**

* O viés é **sempre para mais**, nunca para menos — verificado em teste. Um molde
  ligeiramente maior é recuperável no acabamento; um menor não é.
* Fotografar o mais perpendicular possível é a única mitigação eficaz. O *quality
  gate* mede a inclinação e avisa.
* Larguras (antepé, mediopé, calcâneo) **não** sofrem esse efeito na mesma escala:
  são medidas em regiões sem estrutura fina e permanecem em ~0,2 mm mesmo inclinadas.

---

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
