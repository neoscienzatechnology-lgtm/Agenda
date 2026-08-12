"""Gerador de cenas sintéticas com geometria física conhecida.

Renderiza o plano da plataforma em mm (marcador de 50 × 50 mm + contornos plantares de
comprimento exato) e depois **fotografa** esse plano com uma câmera virtual (posição,
inclinação, rotação, distância focal). O resultado é uma imagem equivalente a uma foto
real, cuja verdade dimensional é conhecida por construção.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..calibration.marker import generate_marker_image
from ..config import get_settings
from . import foot_shape


@dataclass
class FootSpec:
    length_mm: float
    laterality: str = "right"
    center_mm: tuple[float, float] = (0.0, 0.0)
    rotation_deg: float = 0.0


@dataclass
class SceneSpec:
    feet: list[FootSpec] = field(default_factory=list)
    view: str = "below"
    marker_origin_mm: tuple[float, float] = (-25.0, 168.0)
    marker_id: int = 7
    plane_px_per_mm: float = 8.0
    plane_extent_mm: tuple[float, float, float, float] = (-210.0, -200.0, 210.0, 245.0)
    image_size: tuple[int, int] = (2400, 3200)   # (largura, altura)
    camera_distance_mm: float = 700.0
    tilt_deg: float = 0.0
    azimuth_deg: float = 0.0
    roll_deg: float = 0.0
    field_mm: float = 0.0   # 0 = enquadramento automático sobre o conteúdo da cena
    background: str = "light"
    noise_sigma: float = 2.5
    motion_blur_px: int = 0
    defocus_px: int = 0
    glare: float = 0.0
    jpeg_quality: int = 0        # 0 = sem recompressão
    skin_bgr: tuple[int, int, int] = (142, 168, 202)
    seed: int = 12345


@dataclass
class SceneResult:
    image_bgr: np.ndarray
    truth_contours_mm: list[np.ndarray]     # no frame do MARCADOR (origem = canto TL)
    truth_lateralities: list[str]
    truth_lengths_mm: list[float]
    truth_landmarks_mm: list[dict]
    marker_corners_plane_mm: np.ndarray
    homography_plane_to_image: np.ndarray
    spec: SceneSpec

    def encode_png(self) -> bytes:
        ok, buf = cv2.imencode(".png", self.image_bgr)
        if not ok:
            raise RuntimeError("falha ao codificar cena")
        return buf.tobytes()

    def encode_jpeg(self, quality: int = 92) -> bytes:
        ok, buf = cv2.imencode(".jpg", self.image_bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("falha ao codificar cena")
        return buf.tobytes()


def _camera_homography(spec: SceneSpec, center_mm: np.ndarray,
                       content_size_mm: tuple[float, float] | None = None) -> np.ndarray:
    """H que leva (X, Y) do plano em mm para pixels da imagem."""
    w, h = spec.image_size
    if spec.field_mm > 0:
        field_mm = spec.field_mm
    else:
        # Enquadramento automático: garante que todo o conteúdo (pés + marcador +
        # zona de silêncio) caiba na foto com 8 % de folga.
        cw, ch = content_size_mm or (1.0, 1.0)
        field_mm = 1.08 * max(ch, cw * h / max(w, 1))
    f = h * spec.camera_distance_mm / max(field_mm, 1e-6)
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)

    tilt = math.radians(spec.tilt_deg)
    az = math.radians(spec.azimuth_deg)
    # A câmera fica do lado que **vê** o raster do plano sem espelhamento (podoscópio:
    # sob o vidro). O componente z negativo garante que, com tilt = 0, o mapeamento
    # plano→imagem seja a identidade escalada: +x à direita, +y para baixo.
    n_dir = np.array([math.sin(tilt) * math.cos(az),
                      math.sin(tilt) * math.sin(az),
                      -math.cos(tilt)], dtype=np.float64)
    center3 = np.array([center_mm[0], center_mm[1], 0.0], dtype=np.float64)
    cam = center3 + spec.camera_distance_mm * n_dir
    forward = -n_dir

    roll = math.radians(spec.roll_deg)
    up_hint = np.array([math.sin(roll), -math.cos(roll), 0.0], dtype=np.float64)
    if abs(float(np.dot(up_hint, forward))) > 0.995:
        up_hint = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    R = np.stack([right, -true_up, forward], axis=0)   # linhas: x_c, y_c, z_c
    t = -R @ cam
    P = np.stack([R[:, 0], R[:, 1], t], axis=1)        # [r1 r2 t]
    H = K @ P
    return H / H[2, 2]


def _render_plane(spec: SceneSpec, rng: np.random.Generator
                  ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    settings = get_settings()
    x0, y0, x1, y1 = spec.plane_extent_mm
    ppm = spec.plane_px_per_mm
    w = int(round((x1 - x0) * ppm))
    h = int(round((y1 - y0) * ppm))

    if spec.background == "dark":
        base = np.full((h, w, 3), (46, 48, 52), dtype=np.uint8)
    else:
        base = np.full((h, w, 3), (233, 236, 238), dtype=np.uint8)
    base = base.astype(np.float32)
    base += rng.normal(0.0, 1.6, base.shape).astype(np.float32)

    def to_px(pts_mm: np.ndarray) -> np.ndarray:
        return np.stack([(pts_mm[:, 0] - x0) * ppm, (pts_mm[:, 1] - y0) * ppm], axis=1)

    # --- pés ---
    contours_mm: list[np.ndarray] = []
    canvas = base.copy()
    for foot in spec.feet:
        poly_mm = foot_shape.foot_polygon(
            foot.length_mm, foot.laterality, spec.view,
            center_mm=foot.center_mm, rotation_deg=foot.rotation_deg,
        )
        contours_mm.append(poly_mm)
        pts = to_px(poly_mm).astype(np.int32)
        layer = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(layer, [pts], 255, lineType=cv2.LINE_AA)
        alpha = (layer.astype(np.float32) / 255.0)[:, :, None]
        skin = np.array(spec.skin_bgr, dtype=np.float32)[None, None, :]
        texture = rng.normal(0.0, 6.0, (h, w, 1)).astype(np.float32)
        # Leve gradiente radial para simular volume/iluminação da planta.
        canvas = canvas * (1 - alpha) + (skin + texture) * alpha

    # --- marcador de 50 × 50 mm ---
    mx, my = spec.marker_origin_mm
    side_px = int(round(settings.marker_size_mm * ppm))
    marker = generate_marker_image(spec.marker_id, side_px)
    quiet_mm = 12.0
    quiet_px = int(round(quiet_mm * ppm))
    px0 = int(round((mx - x0) * ppm))
    py0 = int(round((my - y0) * ppm))
    qx0, qy0 = px0 - quiet_px, py0 - quiet_px
    qx1, qy1 = px0 + side_px + quiet_px, py0 + side_px + quiet_px
    if qx0 < 0 or qy0 < 0 or qx1 > w or qy1 > h:
        raise ValueError("marcador fora da extensão do plano sintético")
    # A zona de silêncio não pode cobrir nenhum pé — isso invalidaria a verdade
    # geométrica da cena.
    quiet_mm_box = (mx - quiet_mm, my - quiet_mm,
                    mx + settings.marker_size_mm + quiet_mm,
                    my + settings.marker_size_mm + quiet_mm)
    for poly_mm in contours_mm:
        lo, hi = poly_mm.min(axis=0), poly_mm.max(axis=0)
        if (lo[0] < quiet_mm_box[2] and hi[0] > quiet_mm_box[0]
                and lo[1] < quiet_mm_box[3] and hi[1] > quiet_mm_box[1]):
            raise ValueError(
                "zona de silêncio do marcador sobrepõe um pé na cena sintética"
            )
    canvas[qy0:qy1, qx0:qx1] = 250.0
    canvas[py0:py0 + side_px, px0:px0 + side_px] = marker[:, :, None].astype(np.float32)

    marker_corners_plane = np.array([
        [mx, my],
        [mx + settings.marker_size_mm, my],
        [mx + settings.marker_size_mm, my + settings.marker_size_mm],
        [mx, my + settings.marker_size_mm],
    ], dtype=np.float64)

    plane_img = np.clip(canvas, 0, 255).astype(np.uint8)
    return plane_img, marker_corners_plane, contours_mm


def render_scene(spec: SceneSpec) -> SceneResult:
    rng = np.random.default_rng(spec.seed)
    plane_img, marker_corners_plane, contours_mm = _render_plane(spec, rng)

    x0, y0, x1, y1 = spec.plane_extent_mm
    ppm = spec.plane_px_per_mm
    # mm → raster do plano
    A = np.array([[ppm, 0, -x0 * ppm], [0, ppm, -y0 * ppm], [0, 0, 1]], dtype=np.float64)

    quiet = 12.0
    content = np.vstack(
        ([np.vstack(contours_mm)] if contours_mm else [])
        + [marker_corners_plane + np.array([[-quiet, -quiet], [quiet, -quiet],
                                            [quiet, quiet], [-quiet, quiet]])]
    )
    lo, hi = content.min(axis=0), content.max(axis=0)
    scene_center = (lo + hi) / 2.0
    content_size = (float(hi[0] - lo[0]), float(hi[1] - lo[1]))

    H_plane_img = _camera_homography(spec, scene_center, content_size)
    H_raster_img = H_plane_img @ np.linalg.inv(A)

    w, h = spec.image_size
    img = cv2.warpPerspective(plane_img, H_raster_img, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(70, 72, 76))   # piso ao redor da plataforma

    img = img.astype(np.float32)
    if spec.glare > 0:
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
        cxg, cyg = w * 0.62, h * 0.38
        r2 = ((gx - cxg) / (w * 0.16)) ** 2 + ((gy - cyg) / (h * 0.10)) ** 2
        img += (255.0 * spec.glare * np.exp(-r2))[:, :, None]
    if spec.defocus_px > 0:
        k = spec.defocus_px * 2 + 1
        img = cv2.GaussianBlur(img, (k, k), spec.defocus_px * 0.6)
    if spec.motion_blur_px > 0:
        k = max(3, spec.motion_blur_px)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, kernel)
    if spec.noise_sigma > 0:
        img += rng.normal(0.0, spec.noise_sigma, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)

    if spec.jpeg_quality > 0:
        ok, buf = cv2.imencode(".jpg", img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), spec.jpeg_quality])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Verdade no frame do MARCADOR (origem = canto superior esquerdo do marcador).
    origin = marker_corners_plane[0]
    truth = [c - origin for c in contours_mm]
    landmarks = []
    for foot in spec.feet:
        lm = foot_shape.expected_landmarks_mm(
            foot.length_mm, foot.laterality, spec.view,
            center_mm=foot.center_mm, rotation_deg=foot.rotation_deg,
        )
        landmarks.append({k: v - origin for k, v in lm.items()})

    return SceneResult(
        image_bgr=img,
        truth_contours_mm=truth,
        truth_lateralities=[f.laterality for f in spec.feet],
        truth_lengths_mm=[f.length_mm for f in spec.feet],
        truth_landmarks_mm=landmarks,
        marker_corners_plane_mm=marker_corners_plane,
        homography_plane_to_image=H_plane_img,
        spec=spec,
    )


# ------------------------------------------------------------------ cenas prontas


def bilateral_scene(length_left_mm: float = 258.0, length_right_mm: float = 261.0,
                    **overrides) -> SceneSpec:
    """Duas plantas lado a lado, marcador acima e entre elas (layout de podoscópio)."""
    spec = SceneSpec(
        feet=[
            # Vista de baixo: o pé DIREITO aparece à esquerda da imagem.
            FootSpec(length_right_mm, "right", center_mm=(-72.0, -35.0)),
            FootSpec(length_left_mm, "left", center_mm=(72.0, -35.0)),
        ],
    )
    for k, v in overrides.items():
        setattr(spec, k, v)
    return spec


def single_foot_scene(length_mm: float = 265.0, laterality: str = "right",
                      **overrides) -> SceneSpec:
    spec = SceneSpec(
        feet=[FootSpec(length_mm, laterality, center_mm=(0.0, -35.0))],
        image_size=(2000, 2600),
        field_mm=400.0,
    )
    for k, v in overrides.items():
        setattr(spec, k, v)
    return spec
