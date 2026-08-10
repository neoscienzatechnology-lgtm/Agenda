"""Imagem anotada em alta resolução (PNG).

**Esta imagem é para análise visual.** Ela NÃO é o arquivo metrológico de fabricação —
esse papel é exclusivo do PDF A4 1:1. O texto impresso na própria imagem diz isso.

O raster é gerado a partir da geometria em mm com um fator px/mm explícito, de modo
que a imagem também é dimensionalmente coerente (mas continua sendo um raster).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..geometry.polygon import as_xy

BG = (250, 250, 249)
CONTOUR = (74, 33, 11)        # BGR do azul-escuro clínico
AXIS = (170, 155, 140)
METATARSAL = (217, 115, 28)
TOE = (120, 90, 40)
MEDIAL_ARCH = (92, 158, 23)
LATERAL_ARCH = (230, 140, 64)
SUPPORT = (24, 130, 249)
TEXT = (74, 41, 15)
WARN = (60, 60, 220)


class AnnotatedRenderer:
    def __init__(self, px_per_mm: float = 8.0, margin_mm: float = 14.0):
        self.px_per_mm = float(px_per_mm)
        self.margin_mm = float(margin_mm)
        self.origin_mm = np.zeros(2)

    def _px(self, pts_mm) -> np.ndarray:
        return (as_xy(pts_mm) - self.origin_mm) * self.px_per_mm

    def _ipx(self, pts_mm) -> np.ndarray:
        return np.round(self._px(pts_mm)).astype(np.int32)

    def render(self, feet: list[dict], background_bgr: np.ndarray | None = None,
               background_origin_mm: np.ndarray | None = None,
               background_px_per_mm: float | None = None,
               include_callosities: bool = True) -> np.ndarray:
        all_pts = np.vstack([as_xy(f["contourMm"])
                             for f in feet if len(f.get("contourMm", [])) >= 3])
        lo = all_pts.min(axis=0) - self.margin_mm
        hi = all_pts.max(axis=0) + self.margin_mm
        self.origin_mm = lo
        w = int(np.ceil((hi[0] - lo[0]) * self.px_per_mm))
        h = int(np.ceil((hi[1] - lo[1]) * self.px_per_mm)) + int(26 * self.px_per_mm)

        canvas = np.full((h, w, 3), BG, dtype=np.uint8)

        if background_bgr is not None and background_origin_mm is not None \
                and background_px_per_mm:
            canvas = self._blit_background(canvas, background_bgr,
                                           np.asarray(background_origin_mm),
                                           float(background_px_per_mm))

        for foot in feet:
            self._draw_foot(canvas, foot, include_callosities)

        self._draw_footer(canvas, feet)
        return canvas

    # ------------------------------------------------------------------ camadas
    def _blit_background(self, canvas: np.ndarray, bg: np.ndarray,
                         bg_origin_mm: np.ndarray, bg_ppm: float) -> np.ndarray:
        h, w = canvas.shape[:2]
        scale = self.px_per_mm / bg_ppm
        M = np.array([
            [scale, 0.0, (bg_origin_mm[0] - self.origin_mm[0]) * self.px_per_mm],
            [0.0, scale, (bg_origin_mm[1] - self.origin_mm[1]) * self.px_per_mm],
        ], dtype=np.float64)
        warped = cv2.warpAffine(bg, M, (w, h), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=BG)
        mask = cv2.warpAffine(np.full(bg.shape[:2], 255, np.uint8), M, (w, h),
                              flags=cv2.INTER_NEAREST, borderValue=0)
        alpha = (mask.astype(np.float32) / 255.0 * 0.85)[:, :, None]
        return np.clip(canvas * (1 - alpha) + warped * alpha, 0, 255).astype(np.uint8)

    def _draw_foot(self, canvas: np.ndarray, foot: dict, include_callosities: bool
                   ) -> None:
        ppm = self.px_per_mm
        thin = max(1, int(round(0.22 * ppm)))
        thick = max(2, int(round(0.42 * ppm)))

        overlay = canvas.copy()
        for zone in foot.get("supportZones", []) or []:
            poly = as_xy(zone.get("polygonMm"))
            if len(poly) >= 3:
                cv2.fillPoly(overlay, [self._ipx(poly)], SUPPORT, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.16, canvas, 0.84, 0, canvas)

        if include_callosities:
            for hint in foot.get("callosityHints", []) or []:
                poly = as_xy(hint.get("polygonMm"))
                if len(poly) >= 3:
                    cv2.polylines(canvas, [self._ipx(poly)], True, (40, 40, 210),
                                  thin, cv2.LINE_AA)

        for key, color, style in (("medialArch", MEDIAL_ARCH, "dash"),
                                  ("lateralArch", LATERAL_ARCH, "dot")):
            arc = foot.get(key) or {}
            pts = as_xy(arc.get("pointsMm"))
            if len(pts) >= 2:
                self._dashed(canvas, pts, color, thin,
                             on=int(5 * ppm) if style == "dash" else int(2 * ppm),
                             off=int(3 * ppm))

        axis = foot.get("axis")
        if axis:
            pts = np.array([[axis["aMm"]["x"], axis["aMm"]["y"]],
                            [axis["bMm"]["x"], axis["bMm"]["y"]]], dtype=np.float64)
            self._dashed(canvas, pts, AXIS, thin, on=int(6 * ppm), off=int(4 * ppm))

        mt = as_xy(foot.get("metatarsalLineMm"))
        if len(mt) >= 2:
            cv2.polylines(canvas, [self._ipx(mt)], False, METATARSAL, thin, cv2.LINE_AA)

        contour = as_xy(foot.get("contourMm"))
        if len(contour) >= 3:
            cv2.polylines(canvas, [self._ipx(contour)], True, CONTOUR, thick,
                          cv2.LINE_AA)

        for lm in foot.get("landmarks", []) or []:
            pid = lm.get("id", "")
            p = lm.get("positionMm") or {}
            if "x" not in p:
                continue
            q = self._ipx([[p["x"], p["y"]]])[0]
            if pid.startswith("M"):
                color = METATARSAL
                radius = max(2, int(round(0.9 * ppm)))
            elif pid.startswith("T"):
                color = TOE
                radius = max(2, int(round(0.7 * ppm)))
            else:
                color = CONTOUR
                radius = max(2, int(round(0.8 * ppm)))
            cv2.circle(canvas, tuple(q), radius, color, -1, cv2.LINE_AA)
            low = float(lm.get("confidence", 1.0)) < 0.5
            if low:
                cv2.circle(canvas, tuple(q), radius + max(2, int(0.6 * ppm)),
                           WARN, thin, cv2.LINE_AA)
            self._text(canvas, pid, (int(q[0] + 1.6 * ppm), int(q[1] - 1.2 * ppm)),
                       0.36 * ppm / 8.0 * 1.6, color, thin)

        self._draw_labels(canvas, foot)

    def _draw_labels(self, canvas: np.ndarray, foot: dict) -> None:
        contour = as_xy(foot.get("contourMm"))
        if len(contour) < 3:
            return
        ppm = self.px_per_mm
        lo = contour.min(axis=0)
        anchor = self._ipx([[lo[0], lo[1] - 10.0]])[0]
        side = "DIREITO" if foot.get("laterality") == "right" else "ESQUERDO"
        self._text(canvas, side, tuple(anchor), 0.9, TEXT, max(2, int(0.3 * ppm)))

        m = foot.get("measurements") or {}
        rows = [
            f"Comprimento: {m.get('lengthMm', 0):.1f} mm",
            f"Antepé: {m.get('forefootWidthMm', 0):.1f} mm",
            f"Mediopé: {m.get('midfootWidthMm', 0):.1f} mm",
            f"Calcâneo: {m.get('heelWidthMm', 0):.1f} mm",
            f"Calcâneo→MT: {m.get('heelToMetatarsalLineMm', 0):.1f} mm",
            f"Índice de arco (proj.): {m.get('archIndex', 0):.3f}",
        ]
        hi = contour.max(axis=0)
        x = int((hi[0] - self.origin_mm[0]) * ppm) + int(4 * ppm)
        y = int((lo[1] - self.origin_mm[1]) * ppm) + int(4 * ppm)
        for i, row in enumerate(rows):
            self._text(canvas, row, (x, y + i * int(3.6 * ppm)), 0.5, TEXT,
                       max(1, int(0.2 * ppm)))

    def _draw_footer(self, canvas: np.ndarray, feet: list[dict]) -> None:
        h = canvas.shape[0]
        ppm = self.px_per_mm
        self._text(canvas,
                   "Imagem para ANÁLISE VISUAL. O arquivo metrológico de fabricação "
                   "é o PDF A4 em escala 1:1.",
                   (int(3 * ppm), h - int(9 * ppm)), 0.52, WARN,
                   max(1, int(0.22 * ppm)))
        est = any(any(float(lm.get("confidence", 1)) < 0.6
                      and lm.get("id", "").startswith("M")
                      for lm in (f.get("landmarks") or []))
                  for f in feet)
        if est:
            self._text(canvas,
                       "M2–M4 são landmarks ESTIMADOS (não visíveis em foto plantar).",
                       (int(3 * ppm), h - int(4 * ppm)), 0.5, TEXT,
                       max(1, int(0.2 * ppm)))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _text(canvas, txt, org, scale, color, thickness) -> None:
        cv2.putText(canvas, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    thickness, cv2.LINE_AA)

    def _dashed(self, canvas, pts_mm, color, thickness, on: int, off: int) -> None:
        pts = self._px(pts_mm)
        on = max(2, on)
        off = max(2, off)
        acc = 0.0
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            seg = float(np.linalg.norm(b - a))
            if seg < 1e-9:
                continue
            d = a.copy()
            t = 0.0
            while t < seg:
                span = on if (acc % (on + off)) < on else off
                draw = (acc % (on + off)) < on
                step = min(span - (acc % (on + off)) % max(span, 1), seg - t)
                step = max(step, 1.0)
                nxt = a + (b - a) * min(1.0, (t + step) / seg)
                if draw:
                    cv2.line(canvas, tuple(np.round(d).astype(int)),
                             tuple(np.round(nxt).astype(int)), color, thickness,
                             cv2.LINE_AA)
                d = nxt
                t += step
                acc += step
