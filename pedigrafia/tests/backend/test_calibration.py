"""Cadeia metrológica: detecção do marcador, homografia, escala px→mm."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.calibration.homography import (build_rectification, compute_homography,
                                        marker_model_mm, verify_round_trip)
from app.calibration.marker import detect_marker, generate_marker_image
from app.config import get_settings
from app.synth import generator as G


def test_marker_model_is_exactly_50mm():
    model = marker_model_mm(get_settings().marker_size_mm)
    sides = [float(np.linalg.norm(model[(i + 1) % 4] - model[i])) for i in range(4)]
    assert sides == pytest.approx([50.0] * 4, abs=1e-12)


def test_detects_marker_in_clean_render():
    canvas = np.full((900, 900), 255, np.uint8)
    canvas[300:600, 300:600] = generate_marker_image(7, 300)
    det = detect_marker(cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))
    assert det.found and det.marker_id == 7
    assert det.src_px_per_mm == pytest.approx(6.0, abs=0.05)
    assert det.skew < 0.01


def test_missing_marker_is_reported_not_guessed():
    blank = np.full((600, 600, 3), 240, np.uint8)
    det = detect_marker(blank)
    assert not det.found
    assert det.src_px_per_mm == 0.0
    assert "não encontrado" in det.reason


@pytest.mark.parametrize("tilt,azimuth", [(0.0, 0.0), (10.0, 30.0), (20.0, 140.0),
                                          (28.0, 250.0)])
def test_homography_recovers_marker_corners_exactly(scenes, tilt, azimuth):
    scene = scenes(tilt_deg=tilt, azimuth_deg=azimuth)
    det = detect_marker(scene.image_bgr)
    assert det.found
    H, _ = compute_homography(det)
    corners_mm = cv2.perspectiveTransform(
        det.corners_px.reshape(-1, 1, 2), H).reshape(-1, 2)
    assert corners_mm == pytest.approx(marker_model_mm(50.0), abs=1e-6)


@pytest.mark.parametrize("tilt,azimuth", [(0.0, 0.0), (12.0, 35.0), (22.0, 160.0),
                                          (30.0, 300.0)])
def test_rectified_marker_roundtrip_is_50mm(scenes, tilt, azimuth):
    """I1: re-medir o marcador na imagem retificada valida a cadeia inteira."""
    scene = scenes(tilt_deg=tilt, azimuth_deg=azimuth)
    det = detect_marker(scene.image_bgr)
    H, _ = compute_homography(det)
    rect = build_rectification(scene.image_bgr, det, H)
    sides, error = verify_round_trip(rect)
    assert len(sides) == 4
    assert error < 0.3, f"lados re-medidos: {sides}"


def test_px_mm_conversion_is_exactly_invertible(scenes):
    scene = scenes()
    det = detect_marker(scene.image_bgr)
    rect = build_rectification(scene.image_bgr, det)
    pts = np.array([[0.0, 0.0], [123.4, -56.7], [999.9, 4321.0], [-12.0, 8.5]])
    back = rect.mm_to_rect_px(rect.rect_px_to_mm(pts))
    assert back == pytest.approx(pts, abs=1e-9)


def test_rectification_scale_matches_configured_px_per_mm(scenes):
    scene = scenes()
    det = detect_marker(scene.image_bgr)
    rect = build_rectification(scene.image_bgr, det)
    assert rect.px_per_mm == pytest.approx(get_settings().rectified_px_per_mm, rel=1e-9)
    # 50 mm no plano devem ocupar exatamente 50 * px_per_mm pixels no raster.
    a = rect.mm_to_rect_px(np.array([[0.0, 0.0]]))[0]
    b = rect.mm_to_rect_px(np.array([[50.0, 0.0]]))[0]
    assert float(np.linalg.norm(b - a)) == pytest.approx(50.0 * rect.px_per_mm, abs=1e-9)


def test_rectification_window_includes_marker_and_feet(scenes):
    scene = scenes()
    det = detect_marker(scene.image_bgr)
    rect = build_rectification(scene.image_bgr, det)
    x0, y0, x1, y1 = rect.extent_mm
    assert x0 <= 0 and y0 <= 0 and x1 >= 50 and y1 >= 50
    for truth in scene.truth_contours_mm:
        assert truth[:, 0].min() >= x0 - 1 and truth[:, 0].max() <= x1 + 1
        assert truth[:, 1].min() >= y0 - 1 and truth[:, 1].max() <= y1 + 1


def test_scale_is_reproducible_across_independent_captures():
    """Duas capturas diferentes do mesmo objeto não podem dar escalas arbitrárias."""
    lengths = []
    for tilt, az, seed in ((0.0, 0.0, 11), (14.0, 70.0, 22), (24.0, 200.0, 33)):
        scene = G.render_scene(
            G.single_foot_scene(265.0, "right", tilt_deg=tilt, azimuth_deg=az, seed=seed))
        det = detect_marker(scene.image_bgr)
        rect = build_rectification(scene.image_bgr, det)
        sides, _ = verify_round_trip(rect)
        lengths.append(float(np.mean(sides)))
    assert max(lengths) - min(lengths) < 0.25, lengths
    assert all(abs(v - 50.0) < 0.2 for v in lengths), lengths
