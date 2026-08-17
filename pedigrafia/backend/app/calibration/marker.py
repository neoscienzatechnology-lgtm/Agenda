"""Detecção do marcador fiducial de 50 × 50 mm.

O marcador é a **única** referência metrológica do sistema. Nada além dele define
escala física. Suportamos ArUco e AprilTag (ambos disponíveis em ``cv2.aruco``); a
única exigência é que os quatro cantos sejam recuperáveis com estabilidade subpixel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..config import get_settings

# Ordem canônica dos cantos devolvida pelo OpenCV, no referencial do próprio marcador:
#   0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left
CORNER_ORDER = ("TL", "TR", "BR", "BL")


@dataclass
class DetectedMarker:
    """Um marcador individual localizado na foto."""

    marker_id: int
    corners_px: np.ndarray            # (4, 2) float64, ordem TL TR BR BL
    side_lengths_px: tuple
    touches_border: bool

    @property
    def center_px(self) -> np.ndarray:
        return self.corners_px.mean(axis=0)

    @property
    def mean_side_px(self) -> float:
        return float(np.mean(self.side_lengths_px))


@dataclass
class MarkerDetection:
    found: bool
    corners_px: np.ndarray            # (4, 2) float64, ordem TL TR BR BL
    marker_id: int = -1
    dictionary: str = ""
    side_lengths_px: tuple = ()
    src_px_per_mm: float = 0.0
    skew: float = 0.0                 # dispersão relativa dos lados (0 = perfeito)
    angle_deviation_deg: float = 0.0  # maior desvio dos ângulos internos em relação a 90°
    tilt_deg: float = 0.0             # inclinação estimada do plano
    confidence: float = 0.0
    touches_border: bool = False
    reason: str = ""
    markers: tuple = ()
    """Todos os marcadores detectados (``DetectedMarker``), não só o escolhido.

    É a partir desta lista que a calibração multi-marcador monta os pontos de
    controle distribuídos pela plataforma."""
    source: str = "aruco"
    """``aruco`` (marcador impresso) ou ``reference`` (objeto de dimensão normalizada).

    O resto do pipeline trata os dois igualmente — são quatro cantos com posição
    física conhecida. O campo existe para que a interface e os avisos possam dizer
    ao profissional de onde veio a escala."""

    @property
    def detected_ids(self) -> list[int]:
        return [m.marker_id for m in self.markers]

    @property
    def center_px(self) -> np.ndarray:
        return self.corners_px.mean(axis=0)


def _resolve_dictionary(name: str) -> tuple[cv2.aruco.Dictionary, str]:
    canonical = name.strip().upper()
    if not hasattr(cv2.aruco, canonical):
        raise ValueError(f"dicionário de marcador desconhecido: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, canonical)), canonical


def _detector_params() -> cv2.aruco.DetectorParameters:
    params = cv2.aruco.DetectorParameters()
    # Refino subpixel dos cantos: essencial para a precisão metrológica.
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 7
    params.cornerRefinementMaxIterations = 60
    params.cornerRefinementMinAccuracy = 0.01
    # Tolerância maior para fotos de celular com iluminação irregular.
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.045
    return params


def _quad_metrics(corners: np.ndarray) -> tuple[tuple[float, ...], float, float]:
    """Comprimentos dos lados, *skew* relativo e maior desvio angular (graus)."""
    sides = []
    for i in range(4):
        sides.append(float(np.linalg.norm(corners[(i + 1) % 4] - corners[i])))
    mean_side = sum(sides) / 4.0
    skew = (max(sides) - min(sides)) / mean_side if mean_side > 0 else 1.0

    max_dev = 0.0
    for i in range(4):
        a = corners[(i - 1) % 4] - corners[i]
        b = corners[(i + 1) % 4] - corners[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return tuple(sides), 1.0, 90.0
        cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        max_dev = max(max_dev, abs(math.degrees(math.acos(cosang)) - 90.0))
    return tuple(sides), float(skew), float(max_dev)


def estimate_tilt_deg(corners_px: np.ndarray, image_shape: tuple[int, int],
                      marker_size_mm: float,
                      model_mm: Optional[np.ndarray] = None) -> float:
    """Inclinação aproximada do plano do marcador em relação ao plano da imagem.

    Sem calibração intrínseca da câmera assumimos uma distância focal plausível
    (``f ≈ 1.15 · max(dimensão)``, típico de câmeras de celular). O valor resultante é
    **estimado** e usado apenas como sinal de qualidade — jamais para medir.

    ``model_mm`` permite passar um modelo não quadrado (um retângulo de referência,
    por exemplo); omitido, assume o quadrado de ``marker_size_mm``.
    """
    h, w = image_shape[:2]
    f = 1.15 * max(w, h)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)

    if model_mm is None:
        obj = np.array(
            [[0, 0], [marker_size_mm, 0], [marker_size_mm, marker_size_mm],
             [0, marker_size_mm]],
            dtype=np.float64,
        )
    else:
        obj = np.asarray(model_mm, dtype=np.float64).reshape(4, 2)
    try:
        H = cv2.getPerspectiveTransform(obj.astype(np.float32),
                                        corners_px.astype(np.float32))
    except cv2.error:
        return 90.0

    Kinv = np.linalg.inv(K)
    h1 = Kinv @ H[:, 0]
    h2 = Kinv @ H[:, 1]
    n1, n2 = np.linalg.norm(h1), np.linalg.norm(h2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 90.0
    r1 = h1 / n1
    r2 = h2 / n2
    # Ortonormalização de Gram-Schmidt (H nunca é exatamente uma rotação).
    r2 = r2 - np.dot(r1, r2) * r1
    n2b = np.linalg.norm(r2)
    if n2b < 1e-12:
        return 90.0
    r2 = r2 / n2b
    r3 = np.cross(r1, r2)
    tilt = math.degrees(math.acos(min(1.0, abs(float(r3[2])))))
    return float(tilt)


def _touches_frame(quad: np.ndarray, width: int, height: int,
                   margin: float = 2.0) -> bool:
    return bool(
        np.any(quad[:, 0] < margin) or np.any(quad[:, 1] < margin)
        or np.any(quad[:, 0] > width - 1 - margin)
        or np.any(quad[:, 1] > height - 1 - margin)
    )


def _refine_corners(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    side = float(np.linalg.norm(corners[1] - corners[0]))
    win = int(max(3, min(11, round(side / 12.0))))
    pts = corners.astype(np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.005)
    try:
        cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), criteria)
    except cv2.error:
        return corners.astype(np.float64)
    refined = pts.reshape(-1, 2).astype(np.float64)
    # Segurança: rejeita o refino se ele mover um canto de forma implausível.
    if float(np.max(np.linalg.norm(refined - corners, axis=1))) > side * 0.08:
        return corners.astype(np.float64)
    return refined


def detect_marker(bgr: np.ndarray, *, dictionary: Optional[str] = None,
                  marker_id: Optional[int] = None) -> MarkerDetection:
    settings = get_settings()
    dict_name = dictionary or settings.marker_dictionary
    wanted_id = settings.marker_id if marker_id is None else marker_id

    aruco_dict, canonical = _resolve_dictionary(dict_name)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, _detector_params())
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return MarkerDetection(
            found=False, corners_px=np.zeros((4, 2)), dictionary=canonical,
            reason="Marcador de referência não encontrado na imagem.",
        )

    ids_flat = ids.ravel().tolist()
    candidates = list(zip(ids_flat, corners))
    if wanted_id >= 0:
        candidates = [c for c in candidates if c[0] == wanted_id]
        if not candidates:
            return MarkerDetection(
                found=False, corners_px=np.zeros((4, 2)), dictionary=canonical,
                reason=f"Marcador id={wanted_id} não encontrado (vistos: {ids_flat}).",
            )

    # Havendo vários, escolhe o de maior área (o mais próximo/confiável) como
    # referência principal; TODOS entram em `markers` para a calibração distribuída.
    def area(entry):
        pts = entry[1].reshape(4, 2)
        return abs(cv2.contourArea(pts.astype(np.float32)))

    h_img, w_img = bgr.shape[:2]
    all_markers: list[DetectedMarker] = []
    for mid, raw in candidates:
        q = _refine_corners(gray, raw.reshape(4, 2).astype(np.float64))
        sides_i, _, _ = _quad_metrics(q)
        all_markers.append(DetectedMarker(
            marker_id=int(mid), corners_px=q, side_lengths_px=sides_i,
            touches_border=_touches_frame(q, w_img, h_img),
        ))

    chosen_id, chosen = max(candidates, key=area)
    quad = _refine_corners(gray, chosen.reshape(4, 2).astype(np.float64))

    sides, skew, angle_dev = _quad_metrics(quad)
    mean_side = sum(sides) / 4.0
    src_px_per_mm = mean_side / settings.marker_size_mm
    tilt = estimate_tilt_deg(quad, bgr.shape[:2], settings.marker_size_mm)

    touches = _touches_frame(quad, bgr.shape[1], bgr.shape[0])

    # Confiança combinando amostragem, regularidade do quadrilátero e recorte.
    res_term = float(np.clip(src_px_per_mm / 4.0, 0.0, 1.0))
    skew_term = float(np.clip(1.0 - skew / 0.55, 0.0, 1.0))
    ang_term = float(np.clip(1.0 - angle_dev / 42.0, 0.0, 1.0))
    conf = 0.34 * res_term + 0.33 * skew_term + 0.33 * ang_term
    if touches:
        conf *= 0.45

    return MarkerDetection(
        found=True,
        corners_px=quad,
        marker_id=int(chosen_id),
        dictionary=canonical,
        side_lengths_px=sides,
        src_px_per_mm=float(src_px_per_mm),
        skew=skew,
        angle_deviation_deg=angle_dev,
        tilt_deg=tilt,
        confidence=float(np.clip(conf, 0.0, 1.0)),
        touches_border=touches,
        markers=tuple(all_markers),
    )


def generate_marker_image(marker_id: int = 7, side_px: int = 1000,
                          dictionary: Optional[str] = None,
                          border_px: int = 0) -> np.ndarray:
    """Gera a arte do marcador (uso: folha imprimível de calibração)."""
    settings = get_settings()
    aruco_dict, _ = _resolve_dictionary(dictionary or settings.marker_dictionary)
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, side_px)
    if border_px > 0:
        img = cv2.copyMakeBorder(img, border_px, border_px, border_px, border_px,
                                 cv2.BORDER_CONSTANT, value=255)
    return img
