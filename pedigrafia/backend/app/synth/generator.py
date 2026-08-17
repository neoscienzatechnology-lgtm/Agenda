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
class ReferenceSpec:
    """Objeto retangular de dimensão normalizada largado sobre a plataforma."""

    key: str = "card"
    origin_mm: tuple[float, float] = (0.0, 0.0)
    rotation_deg: float = 0.0
    bgr: tuple[int, int, int] = (176, 128, 62)


@dataclass
class SceneSpec:
    feet: list[FootSpec] = field(default_factory=list)
    view: str = "below"
    marker_origin_mm: tuple[float, float] = (-25.0, 168.0)
    marker_id: int = 7
    references: list = field(default_factory=list)
    """``ReferenceSpec``s. Quando presentes, nenhum marcador ArUco é desenhado — a
    cena passa a testar exatamente o caminho de calibração por objeto conhecido."""
    target: object = None
    """Alvo de calibração multi-marcador. Quando definido, ``marker_origin_mm`` e
    ``marker_id`` são ignorados e os marcadores são desenhados nas posições físicas
    declaradas pelo alvo."""
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
                  ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
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
        # `astype(int32)` truncaria em direção a zero e introduziria até 1 px de
        # viés sistemático no contorno de VERDADE. `shift` preserva sub-pixel.
        shift = 4
        pts = np.round(to_px(poly_mm) * (1 << shift)).astype(np.int32)
        layer = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(layer, [pts], 255, lineType=cv2.LINE_AA, shift=shift)
        alpha = (layer.astype(np.float32) / 255.0)[:, :, None]
        skin = np.array(spec.skin_bgr, dtype=np.float32)[None, None, :]
        texture = rng.normal(0.0, 6.0, (h, w, 1)).astype(np.float32)
        # Leve gradiente radial para simular volume/iluminação da planta.
        canvas = canvas * (1 - alpha) + (skin + texture) * alpha

    # --- objetos de referência de dimensão normalizada ---
    if spec.references:
        from ..calibration.reference import resolve_reference

        all_corners: list[np.ndarray] = []
        for item in spec.references:
            ref = resolve_reference(item.key)
            corners = _reference_corners_mm(ref, item)
            for poly_mm in contours_mm:
                if _polygons_intersect(corners, poly_mm):
                    raise ValueError(
                        f"objeto de referência {item.key} sobrepõe um pé na cena")
            outline = _rounded_rect_mm(corners, ref.corner_radius_mm)
            pts = np.round(to_px(outline) * (1 << 4)).astype(np.int32)
            layer = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(layer, [pts], 255, lineType=cv2.LINE_AA, shift=4)
            alpha = (layer.astype(np.float32) / 255.0)[:, :, None]
            body = np.array(item.bgr, dtype=np.float32)[None, None, :]
            body = body + rng.normal(0.0, 3.0, (h, w, 1)).astype(np.float32)
            canvas = canvas * (1 - alpha) + body * alpha
            all_corners.append(corners)

        plane_img = np.clip(canvas, 0, 255).astype(np.uint8)
        return (plane_img, all_corners[0], contours_mm, np.vstack(all_corners))

    # --- marcadores de calibração ---
    quiet_mm = 12.0
    quiet_px = int(round(quiet_mm * ppm))

    if spec.target is not None:
        placements = [(m.marker_id, m.origin_mm, m.size_mm) for m in spec.target.markers]
    else:
        placements = [(spec.marker_id, spec.marker_origin_mm, settings.marker_size_mm)]

    first_corners = None
    all_marker_corners: list[np.ndarray] = []
    for mid, (mx, my), size_mm in placements:
        side_px = int(round(size_mm * ppm))
        marker = generate_marker_image(mid, side_px)
        px0 = int(round((mx - x0) * ppm))
        py0 = int(round((my - y0) * ppm))
        qx0, qy0 = px0 - quiet_px, py0 - quiet_px
        qx1, qy1 = px0 + side_px + quiet_px, py0 + side_px + quiet_px
        if qx0 < 0 or qy0 < 0 or qx1 > w or qy1 > h:
            raise ValueError(f"marcador {mid} fora da extensão do plano sintético")
        # A zona de silêncio não pode cobrir nenhum pé — isso invalidaria a verdade
        # geométrica da cena.
        # Sobreposição REAL entre a zona de silêncio e o pé — comparar caixas
        # envolventes daria falso positivo com marcadores nos cantos da área.
        qx_lo, qy_lo = mx - quiet_mm, my - quiet_mm
        qx_hi, qy_hi = mx + size_mm + quiet_mm, my + size_mm + quiet_mm
        for poly_mm in contours_mm:
            inside_quiet = np.any(
                (poly_mm[:, 0] >= qx_lo) & (poly_mm[:, 0] <= qx_hi)
                & (poly_mm[:, 1] >= qy_lo) & (poly_mm[:, 1] <= qy_hi))
            corners_in_foot = any(
                cv2.pointPolygonTest(poly_mm.astype(np.float32), (float(cx), float(cy)),
                                     False) >= 0
                for cx, cy in ((qx_lo, qy_lo), (qx_hi, qy_lo),
                               (qx_hi, qy_hi), (qx_lo, qy_hi)))
            if inside_quiet or corners_in_foot:
                raise ValueError(
                    f"zona de silêncio do marcador {mid} sobrepõe um pé na cena"
                )
        canvas[qy0:qy1, qx0:qx1] = 250.0
        canvas[py0:py0 + side_px, px0:px0 + side_px] = \
            marker[:, :, None].astype(np.float32)
        corners = np.array([
            [mx, my], [mx + size_mm, my],
            [mx + size_mm, my + size_mm], [mx, my + size_mm],
        ], dtype=np.float64)
        all_marker_corners.append(corners)
        if first_corners is None:
            first_corners = corners

    marker_corners_plane = first_corners

    plane_img = np.clip(canvas, 0, 255).astype(np.uint8)
    return (plane_img, marker_corners_plane, contours_mm,
            np.vstack(all_marker_corners))


def _reference_corners_mm(ref, item) -> np.ndarray:
    """Cantos físicos TL, TR, BR, BL do objeto, já posicionado e girado."""
    a = math.radians(item.rotation_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return ref.model_mm @ R.T + np.array(item.origin_mm, dtype=np.float64)


def _rounded_rect_mm(corners_mm: np.ndarray, radius_mm: float,
                     arc_steps: int = 10) -> np.ndarray:
    """Polígono denso com os cantos arredondados no raio da norma (ID-1: 3,18 mm).

    O cartão físico não tem vértice: quem quiser o canto ideal precisa ajustar as
    retas das arestas e intersectá-las. Renderizar o arredondamento de verdade é o
    que torna esse requisito visível no teste."""
    if radius_mm <= 0.0:
        return corners_mm
    out: list[np.ndarray] = []
    for i in range(4):
        here = corners_mm[i]
        prev_c = corners_mm[(i - 1) % 4]
        next_c = corners_mm[(i + 1) % 4]
        u = (next_c - here) / max(np.linalg.norm(next_c - here), 1e-9)
        v = (prev_c - here) / max(np.linalg.norm(prev_c - here), 1e-9)
        centre = here + radius_mm * (u + v)
        start, end = here + radius_mm * v, here + radius_mm * u
        a0 = math.atan2(*(start - centre)[::-1])
        a1 = math.atan2(*(end - centre)[::-1])
        # Percorre o arco no sentido curto, o que mantém o polígono simples.
        while a1 - a0 > math.pi:
            a1 -= 2 * math.pi
        while a0 - a1 > math.pi:
            a1 += 2 * math.pi
        for t in np.linspace(a0, a1, arc_steps):
            out.append(centre + radius_mm * np.array([math.cos(t), math.sin(t)]))
    return np.array(out, dtype=np.float64)


def _polygons_intersect(a: np.ndarray, b: np.ndarray) -> bool:
    poly_b = b.astype(np.float32)
    if any(cv2.pointPolygonTest(poly_b, (float(x), float(y)), False) >= 0
           for x, y in a):
        return True
    poly_a = a.astype(np.float32)
    return any(cv2.pointPolygonTest(poly_a, (float(x), float(y)), False) >= 0
               for x, y in b)


def render_scene(spec: SceneSpec) -> SceneResult:
    rng = np.random.default_rng(spec.seed)
    plane_img, marker_corners_plane, contours_mm, all_corners = _render_plane(spec, rng)

    x0, y0, x1, y1 = spec.plane_extent_mm
    ppm = spec.plane_px_per_mm
    # mm → raster do plano
    A = np.array([[ppm, 0, -x0 * ppm], [0, ppm, -y0 * ppm], [0, 0, 1]], dtype=np.float64)

    # O enquadramento precisa conter TODOS os marcadores do alvo — senão parte deles
    # sai da foto e a calibração volta a depender de um só.
    quiet = 12.0
    content = np.vstack(
        ([np.vstack(contours_mm)] if contours_mm else [])
        + [all_corners + np.array([[-quiet, -quiet]]),
           all_corners + np.array([[quiet, quiet]])]
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

    # Verdade no frame da REFERÊNCIA: origem no seu canto TL, eixo x ao longo da
    # aresta TL→TR. Para um marcador ArUco alinhado isso é a identidade; para um
    # cartão largado torto é a mesma convenção que `fit_free_rectangles` adota, e
    # sem ela a verdade e a medida ficariam em frames diferentes.
    origin = marker_corners_plane[0]
    ex = marker_corners_plane[1] - origin
    ey = marker_corners_plane[3] - origin
    ex = ex / max(float(np.linalg.norm(ex)), 1e-12)
    ey = ey / max(float(np.linalg.norm(ey)), 1e-12)
    basis = np.stack([ex, ey], axis=1)          # colunas: eixos do frame

    def to_reference_frame(pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, dtype=np.float64).reshape(-1, 2) - origin) @ basis

    truth = [to_reference_frame(c) for c in contours_mm]
    landmarks = []
    for foot in spec.feet:
        lm = foot_shape.expected_landmarks_mm(
            foot.length_mm, foot.laterality, spec.view,
            center_mm=foot.center_mm, rotation_deg=foot.rotation_deg,
        )
        landmarks.append({k: to_reference_frame(v)[0] for k, v in lm.items()})

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


def board_scene(length_left_mm: float = 258.0, length_right_mm: float = 261.0,
                area_mm: tuple[float, float] = (320.0, 480.0),
                **overrides) -> SceneSpec:
    """Cena com o alvo de QUATRO marcadores ao redor da área de apoio.

    Layout recomendado: os pés ficam dentro do quadrilátero dos marcadores, de modo
    que a homografia interpole em vez de extrapolar.
    """
    from ..calibration.target import board4_target

    target = board4_target(area_mm=area_mm)
    w, h = area_mm
    cx, cy = w / 2.0, h / 2.0
    spec = SceneSpec(
        feet=[
            FootSpec(length_right_mm, "right", center_mm=(cx - 62.0, cy)),
            FootSpec(length_left_mm, "left", center_mm=(cx + 62.0, cy)),
        ],
        target=target,
        plane_extent_mm=(-70.0, -70.0, w + 70.0, h + 70.0),
    )
    for k, v in overrides.items():
        setattr(spec, k, v)
    return spec


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


def reference_scene(length_left_mm: float = 258.0, length_right_mm: float = 261.0,
                    key: str = "card", count: int = 1,
                    rotation_deg: float = 11.0, **overrides) -> SceneSpec:
    """Cena calibrada por objeto de dimensão normalizada, sem nenhum ArUco.

    ``count=1`` põe o objeto de um lado só (extrapolação, o caso realista de quem
    não quer imprimir nada); ``count=2`` põe um de cada lado dos pés, que é a
    configuração que devolve a interpolação e a exatidão do tabuleiro.
    """
    from ..calibration.reference import resolve_reference

    ref = resolve_reference(key)

    def place(centre_mm, deg):
        """`origin_mm` é o canto TL; posicionar pelo centro é o que a pessoa faz."""
        a = math.radians(deg)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        half = R @ np.array([ref.width_mm / 2.0, ref.height_mm / 2.0])
        return (float(centre_mm[0] - half[0]), float(centre_mm[1] - half[1]))

    # Objeto com o lado maior quase vertical, encostado ao lado dos pés — que é
    # como um profissional apoiaria um cartão na plataforma.
    upright = 90.0 - rotation_deg
    a = math.radians(upright)
    bbox_half_x = (abs(ref.width_mm / 2.0 * math.cos(a))
                   + abs(ref.height_mm / 2.0 * math.sin(a)))
    reach = 128.0 + bbox_half_x          # 128 mm livres a partir do eixo dos pés
    # Dois objetos vão em diagonal — um na altura dos pododáctilos, outro na do
    # calcâneo. Quatro fecham os cantos, e aí o casco dos pontos de controle envolve
    # os pés: é a mesma condição de interpolação do alvo impresso.
    spots = [(-reach, -118.0), (reach, 52.0), (reach, -118.0), (-reach, 52.0)]
    tints = [(176, 128, 62), (96, 104, 168), (104, 148, 96), (150, 110, 150)]
    items = []
    for i in range(max(1, min(count, 4))):
        cx, cy = spots[i]
        deg = upright if cx < 0 else -upright
        items.append(ReferenceSpec(key=key, origin_mm=place((cx, cy), deg),
                                   rotation_deg=deg, bgr=tints[i]))

    # A plataforma tem de cobrir todo o enquadramento automático da câmera: se
    # sobrar piso escuro demais na foto, o teste passa a medir o gerador em vez do
    # sistema.
    half_x = reach + bbox_half_x + 60.0
    spec = SceneSpec(
        feet=[
            FootSpec(length_right_mm, "right", center_mm=(-62.0, -35.0)),
            FootSpec(length_left_mm, "left", center_mm=(62.0, -35.0)),
        ],
        references=items,
        plane_extent_mm=(-half_x, -1.35 * half_x, half_x, 1.35 * half_x),
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
