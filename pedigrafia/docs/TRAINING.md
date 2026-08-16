# Treinar um modelo específico de segmentação plantar

O sistema funciona ponta a ponta com segmentação clássica (OpenCV). Nada no pipeline
precisa mudar para trocar por um modelo treinado — a interface já está isolada.

> **Antes de treinar, leia isto.** A maior fonte de erro do sistema **não era** a
> segmentação: era a extrapolação da homografia a partir de um marcador único
> (`METROLOGY_CALIBRATION.md`). Corrigido isso com o alvo de quatro marcadores, o erro
> de comprimento caiu para **−0,064 mm em média, 0,137 mm no pior caso** em 10 pés
> sintéticos com inclinação até 24°, rolagem ±15° e distância de 520 a 1050 mm.
>
> Ou seja: **em cena sintética não sobra erro para um modelo corrigir.** O valor de
> treinar um modelo está inteiramente na robustez em **foto real** — pele, sombra,
> meia, esmalte, pelo, patologia, podoscópio sujo — que o gerador não representa.
> Treinar contra dados sintéticos e comemorar a métrica seria enganar a si mesmo.

## 0. Ferramentas já prontas

```bash
pip install -e ./ml-or-vision[train]

# 1. conjunto sintético (máscaras exatas, com anti-aliasing na borda)
python -m pedigrafia_vision.dataset            # via generate_synthetic_dataset()

# 2. treino (U-Net, perda com peso extra na borda, rótulos suaves)
python -m pedigrafia_vision.train --data ./dataset --epochs 40 --out modelo.pt

# 3. exportação (saída = PROBABILIDADE, não máscara binária)
python -m pedigrafia_vision.export_onnx --checkpoint modelo.pt --out modelo.onnx

# 4. critério de aceite EM MILÍMETROS, através do pipeline real
python -m pedigrafia_vision.benchmark --scenes 60 --model modelo.onnx
```

## 1. Onde o modelo entra

```
app/segmentation/base.py       Protocol Segmenter (contrato)
app/segmentation/classical.py  implementação atual (padrão)
app/segmentation/onnx_model.py ponte para qualquer modelo ONNX
ml-or-vision/pedigrafia_vision/onnx_runner.py   execução (sigmoid/softmax, resize)
```

Ativação:

```bash
pip install -e ./ml-or-vision[onnx]
export PEDIGRAFIA_SEGMENTER=onnx
export PEDIGRAFIA_SEG_MODEL=/models/pes-unet.onnx
```

Se o modelo não carregar, o sistema **registra o motivo e cai no clássico** — nunca
devolve resultado vazio silenciosamente (`test_segmenter_is_swappable`).

## 2. Que sinal o modelo precisa devolver

Não basta uma máscara binária. O refino sub-pixel do contorno usa um **campo escalar
contínuo** e a isolinha de 50 %. Um modelo que devolva apenas 0/1 perde ~0,3 mm de
exatidão na fronteira.

* Exporte o mapa de **probabilidade** (antes do `argmax`).
* `OnnxSegmenter` já o publica em `SegmentationResult.score_field`, com
  `score_threshold = 0.5` — a isolinha de meia-probabilidade coincide com a borda
  física se o modelo for treinado com rótulos de borda consistentes.
* Treine com *soft labels* nas bordas (anti-aliasing da máscara de anotação) se
  quiser que a probabilidade seja proporcional à cobertura do pixel. É isso que
  transforma o modelo em um estimador de borda sub-pixel.

## 3. Dados

Necessário fotografar em condições reais de podoscópio:

* **Diversidade obrigatória:** tons de pele (escala de Fitzpatrick I–VI), idade,
  pés com deformidades (hálux valgo, dedos em garra, pé cavo/plano), unhas pintadas,
  pelos, cicatrizes, tatuagens, próteses parciais, amputações de pododáctilos.
* **Variação de captura:** podoscópios claros e retroiluminados, com e sem reflexo,
  celular na mão e em suporte, distâncias de 400 a 1200 mm, inclinação 0–25°.
* **Anotação:** contorno poligonal denso no **raster retificado** (não na foto
  original) — assim o modelo aprende no mesmo espaço em que roda em produção.
* Volume inicial razoável: 800–1500 pés anotados; 2 anotadores com medição de
  concordância entre eles (a variabilidade humana é o piso do erro alcançável).

**Privacidade:** imagens de pés são dado de saúde. Consentimento informado por
escrito, armazenamento segregado, e o conjunto de treino **nunca** deve passar pelo
diretório de sessão do MVP.

## 4. Arquitetura sugerida

| Opção | Quando |
|---|---|
| U-Net / DeepLabV3+ (ResNet-34) | Primeira escolha: uma classe, bordas precisas, CPU viável |
| YOLOv8/v11-seg | Se precisar de instâncias (separar dois pés) sem componentes conexos |
| SAM/SAM2 como auxiliar de anotação | Acelera a rotulagem; **não** use em produção (bordas não calibradas) |

Treine em 768–1024 px de lado. Perda: Dice + BCE, com peso extra numa faixa de ±3 px
ao redor da borda — é a borda que vira milímetro.

## 5. Como saber se o modelo melhorou

Não use IoU como critério de aceite. IoU é insensível justamente ao que importa aqui.

Métricas de aceite, nesta ordem:

1. **Erro de comprimento em mm** contra medição de referência (paquímetro).
2. **Desvio p95 do contorno** em mm contra a anotação humana.
3. **Fidelidade da região dos pododáctilos** — a limitação dominante hoje
   (ver `LIMITATIONS.md` §2). Meça separadamente por região anatômica.
4. IoU/Dice apenas como sinal de regressão grosseira.

O arcabouço para isso já existe: `tests/backend/test_pipeline_metrology.py` roda o
pipeline inteiro contra verdade conhecida. Aponte-o para um conjunto real anotado e
ele vira o *benchmark* do modelo.

## 6. O que o modelo **não** pode fazer

Nem o modelo atual nem qualquer modelo futuro pode influenciar:

* escala física (vem do marcador),
* comprimento final,
* dimensões do molde,
* tamanho do PDF.

O modelo decide **onde está o pé**. A geometria determinística calibrada decide
**quanto ele mede**. Essa separação é arquitetural e está verificada por teste.
