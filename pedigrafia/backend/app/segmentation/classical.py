"""Segmentador clássico (OpenCV) — padrão do MVP.

Estratégia de múltiplas pistas, robusta a podoscópios claros e escuros:

1. modelo de fundo estimado na moldura da área válida (mediana em Lab);
2. mapa de distância cromática ao fundo, em resolução plena (a fronteira precisa ser
   subpixel, então nunca é calculada em escala reduzida);
3. limiar de Otsu sobre esse mapa, com seleção automática de polaridade;
4. prior de pele (YCrCb) como reforço, nunca como requisito — funciona em qualquer
   tom de pele porque só soma evidência;
5. GrabCut em escala reduzida apenas para **filtrar** componentes espúrios (sombras,
   objetos), jamais para definir a fronteira;
6. limpeza morfológica com kernels dimensionados em **milímetros**.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from ..geometry import contour as contour_mod
from ..geometry.shape import shape_stats
from .base import SegmentationContext, SegmentationResult

# Faixa clássica de crominância de pele (invariante a luminância → funciona para
# tons claros e escuros). Usada apenas como evidência adicional.
_SKIN_CR = (133, 183)
_SKIN_CB = (77, 133)


def _background_model(lab: np.ndarray, search: np.ndarray, px_per_mm: float
                      ) -> np.ndarray:
    """Mediana Lab da moldura da região de busca (o fundo domina a borda)."""
    h, w = search.shape
    ring = np.zeros_like(search)
    band = max(3, int(round(8.0 * px_per_mm)))
    ring[:band, :] = 255
    ring[-band:, :] = 255
    ring[:, :band] = 255
    ring[:, -band:] = 255
    ring = cv2.bitwise_and(ring, search)
    if int(np.count_nonzero(ring)) < 200:
        ring = search
    px = lab[ring > 0]
    if len(px) == 0:
        return np.array([200.0, 128.0, 128.0], dtype=np.float32)
    return np.median(px.astype(np.float32), axis=0)


def _skin_evidence(bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1].astype(np.int16)
    cb = ycrcb[:, :, 2].astype(np.int16)
    inside = ((cr >= _SKIN_CR[0]) & (cr <= _SKIN_CR[1])
              & (cb >= _SKIN_CB[0]) & (cb <= _SKIN_CB[1]))
    return inside.astype(np.float32)


def _otsu_on(values: np.ndarray) -> tuple[float, float]:
    """Limiar de Otsu e separabilidade (variância entre classes normalizada)."""
    v = values[np.isfinite(values)]
    if v.size < 64:
        return float(np.mean(v) if v.size else 0.0), 0.0
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-6:
        return lo, 0.0
    scaled = np.clip((v - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr_val = lo + (thr / 255.0) * (hi - lo)
    below = v[v <= thr_val]
    above = v[v > thr_val]
    if below.size == 0 or above.size == 0:
        return thr_val, 0.0
    w0 = below.size / v.size
    w1 = above.size / v.size
    between = w0 * w1 * (float(np.mean(above)) - float(np.mean(below))) ** 2
    total = float(np.var(v))
    sep = between / total if total > 1e-9 else 0.0
    return float(thr_val), float(np.clip(sep, 0.0, 1.0))


def _morph_mm(mask: np.ndarray, mm: float, px_per_mm: float, op: int) -> np.ndarray:
    k = int(round(mm * px_per_mm))
    if k < 1:
        return mask
    k = 2 * k + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, op, kernel)


def _grabcut_filter(bgr: np.ndarray, mask: np.ndarray, search: np.ndarray,
                    px_per_mm: float) -> np.ndarray | None:
    """GrabCut em escala reduzida. Devolve máscara FG grosseira (ou None se falhar)."""
    h, w = mask.shape
    target = 480.0
    scale = min(1.0, target / max(h, w))
    if scale <= 0 or min(int(h * scale), int(w * scale)) < 60:
        return None
    small_bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, (small_bgr.shape[1], small_bgr.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    small_search = cv2.resize(search, (small_bgr.shape[1], small_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    erode_px = max(1, int(round(6.0 * px_per_mm * scale)))
    dilate_px = max(2, int(round(10.0 * px_per_mm * scale)))
    sure_fg = cv2.erode(small_mask, np.ones((erode_px, erode_px), np.uint8))
    maybe = cv2.dilate(small_mask, np.ones((dilate_px, dilate_px), np.uint8))

    gc = np.full(small_mask.shape, cv2.GC_BGD, dtype=np.uint8)
    gc[maybe > 0] = cv2.GC_PR_BGD
    gc[small_mask > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD
    gc[small_search == 0] = cv2.GC_BGD

    if not np.any(gc == cv2.GC_FGD) or not np.any(gc == cv2.GC_BGD):
        return None
    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(small_bgr, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)


class ClassicalSegmenter:
    name = "classical"

    def segment(self, rect_bgr: np.ndarray,
                ctx: SegmentationContext) -> SegmentationResult:
        notes: list[str] = []
        debug: dict[str, np.ndarray] = {}

        search = cv2.bitwise_and(ctx.valid_mask,
                                 cv2.bitwise_not(ctx.exclude_mask))
        if int(np.count_nonzero(search)) < 500:
            return SegmentationResult(
                mask=np.zeros(rect_bgr.shape[:2], np.uint8), confidence=0.0,
                method=self.name, notes=["área de busca vazia"])

        blurred = cv2.GaussianBlur(rect_bgr, (0, 0), max(0.6, 0.18 * ctx.px_per_mm))
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)
        bg = _background_model(lab, search, ctx.px_per_mm)

        diff = lab.astype(np.float32) - bg[None, None, :]
        # Crominância pesa mais que luminância: sombras alteram L mas não a*/b*.
        dist = np.sqrt(0.45 * diff[:, :, 0] ** 2
                       + 1.6 * diff[:, :, 1] ** 2
                       + 1.6 * diff[:, :, 2] ** 2)

        skin = _skin_evidence(blurred)
        # A pele reforça a distância sem jamais ser condição necessária.
        #
        # ATENÇÃO: `score` serve para CLASSIFICAR (o que é pé). Ele NÃO serve para
        # localizar a borda, porque o reforço multiplicativo da pele liga/desliga
        # dentro da faixa de transição e destrói a monotonicidade em relação à
        # cobertura do pixel. A fronteira geométrica é refinada sobre `dist`, que é
        # aproximadamente linear na fração de cobertura.
        score = dist * (1.0 + 0.35 * skin)
        debug["score"] = np.clip(score, 0, 255).astype(np.uint8)

        inside = search > 0
        thr, separability = _otsu_on(score[inside])
        raw = np.zeros(score.shape, np.uint8)
        raw[(score > thr) & inside] = 255

        frac = float(np.count_nonzero(raw)) / float(np.count_nonzero(inside))
        if frac > 0.62:
            # Polaridade invertida: o "fundo" amostrado era na verdade o pé.
            raw = np.zeros(score.shape, np.uint8)
            raw[(score <= thr) & inside] = 255
            notes.append("polaridade invertida automaticamente")

        raw = _morph_mm(raw, 0.8, ctx.px_per_mm, cv2.MORPH_OPEN)
        raw = _morph_mm(raw, 1.6, ctx.px_per_mm, cv2.MORPH_CLOSE)
        raw = (ndimage.binary_fill_holes(raw > 0).astype(np.uint8)) * 255

        gc = _grabcut_filter(rect_bgr, raw, search, ctx.px_per_mm)
        if gc is not None:
            debug["grabcut"] = gc

        kept, dropped = self._select_components(raw, gc, ctx)
        if dropped:
            notes.append(f"{dropped} componente(s) descartado(s) por área/forma")

        contrast = float(np.mean(score[kept > 0]) - np.mean(score[(search > 0) & (kept == 0)])) \
            if np.any(kept > 0) and np.any((search > 0) & (kept == 0)) else 0.0

        n_found = int(cv2.connectedComponents((kept > 0).astype(np.uint8))[0] - 1)
        area_ok = 1.0 if n_found > 0 else 0.0
        confidence = float(np.clip(
            0.55 * separability + 0.25 * min(1.0, contrast / 45.0) + 0.20 * area_ok,
            0.0, 1.0))

        edge_level = self._edge_level(dist, kept, search, float(thr), ctx)
        return SegmentationResult(mask=kept, confidence=confidence, method=self.name,
                                  notes=notes, debug=debug,
                                  score_field=dist, score_threshold=edge_level)

    @staticmethod
    def _edge_level(dist: np.ndarray, mask: np.ndarray, search: np.ndarray,
                    otsu_threshold: float, ctx: SegmentationContext) -> float:
        """Nível do campo de distância que corresponde à fronteira geométrica real.

        O limiar de Otsu separa as *classes* (serve para decidir o que é pé), mas não
        marca a posição da borda: ele cai entre as médias das classes, o que em uma
        transição suavizada (foto rotacionada, reamostrada ou levemente desfocada)
        fica **fora** do objeto e dilata o contorno em mais de 1 mm.

        A fronteira de uma borda com meia-cobertura é a isolinha de 50 % entre o
        nível do fundo e o nível do interior. Como `dist` é aproximadamente linear na
        fração de cobertura do pixel, essa isolinha coincide com a borda física.
        """
        interior = cv2.erode(
            (mask > 0).astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * max(1, int(round(2.0 * ctx.px_per_mm))) + 1,) * 2),
        )
        background = (search > 0) & (mask == 0)
        if int(np.count_nonzero(interior)) < 200 or int(np.count_nonzero(background)) < 200:
            return float(otsu_threshold)
        fg_level = float(np.median(dist[interior > 0]))
        bg_level = float(np.median(dist[background]))
        if not np.isfinite(fg_level) or not np.isfinite(bg_level) or fg_level <= bg_level:
            return float(otsu_threshold)
        return 0.5 * (fg_level + bg_level)

    # ------------------------------------------------------------------ componentes
    def _select_components(self, mask: np.ndarray, gc: np.ndarray | None,
                           ctx: SegmentationContext) -> tuple[np.ndarray, int]:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), connectivity=8)
        out = np.zeros_like(mask)
        min_px = ctx.mm2_to_px(ctx.min_area_mm2)
        max_px = ctx.mm2_to_px(ctx.max_area_mm2)
        dropped = 0
        candidates: list[tuple[float, int]] = []

        for i in range(1, num):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < min_px or area > max_px:
                dropped += 1
                continue
            comp = (labels == i)
            if gc is not None:
                overlap = float(np.count_nonzero(comp & (gc > 0))) / area
                if overlap < 0.35:
                    dropped += 1
                    continue

            # Plausibilidade de forma, avaliada em milímetros sobre o contorno real.
            single = np.where(comp, 255, 0).astype(np.uint8)
            cpx = contour_mod.largest_subpixel_contour_px(single)
            if cpx is None or len(cpx) < 16:
                dropped += 1
                continue
            contour_mm = ctx.origin_mm + cpx / ctx.px_per_mm
            shp = shape_stats(contour_mm)
            if shp.likeness < 0.12:
                dropped += 1
                continue
            touching = self._touches_border(comp, ctx)
            candidates.append((0 if touching else 1, shp.likeness * shp.area_mm2, i))

        # No máximo dois pés; mantém os mais plausíveis (não simplesmente os maiores).
        # Componentes que encostam na borda da área válida (moldura do podoscópio,
        # piso, sombras da plataforma) perdem para qualquer candidato inteiro — mas
        # são preservados quando são a única evidência de pé, para que o *quality
        # gate* possa avisar "pé cortado" em vez de o sistema dizer "nenhum pé".
        candidates.sort(reverse=True)
        keep = candidates[:2]
        if any(c[0] == 1 for c in keep):
            keep = [c for c in keep if c[0] == 1]
        for _, _, i in keep:
            out[labels == i] = 255
        dropped += max(0, len(candidates) - len(keep))
        return out, dropped

    @staticmethod
    def _touches_border(comp: np.ndarray, ctx: SegmentationContext) -> bool:
        """O componente encosta na fronteira da região válida da retificação?"""
        border = cv2.morphologyEx(
            (ctx.valid_mask > 0).astype(np.uint8),
            cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
        return bool(np.any(comp & (border > 0)))
