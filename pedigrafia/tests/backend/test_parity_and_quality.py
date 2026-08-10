"""Paridade Python↔TypeScript e comportamento do quality gate."""

from __future__ import annotations

import numpy as np
import pytest

from app.geometry.frame import frame_from_axis_points
from app.image_io import DecodedImage
from app.measurements.compute import compute_measurements
from app.pipeline import PipelineBlocked, analyze
from app.synth import generator as G


def to_np(points) -> np.ndarray:
    return np.array([[p["x"], p["y"]] for p in points], dtype=np.float64)


def test_parity_fixture_reproduces_in_python(parity_fixture):
    """O lado Python do *fixture* de paridade (o lado TS roda no vitest)."""
    tol = parity_fixture["toleranceMm"]
    for case in parity_fixture["cases"]:
        data = case["input"]
        expected = case["expected"]
        contour = to_np(data["contourMm"])
        a = np.array([data["axis"]["aMm"]["x"], data["axis"]["aMm"]["y"]])
        b = np.array([data["axis"]["bMm"]["x"], data["axis"]["bMm"]["y"]])
        m1 = np.array([data["m1Mm"]["x"], data["m1Mm"]["y"]])
        m5 = np.array([data["m5Mm"]["x"], data["m5Mm"]["y"]])

        frame = frame_from_axis_points(contour, a, b, 1)
        meas, _ = compute_measurements(contour, frame, m1, m5, data["toeCutT"])

        assert meas.length_mm == pytest.approx(expected["lengthMm"], abs=tol), case["name"]
        assert meas.forefoot_width_mm == pytest.approx(expected["forefootWidthMm"], abs=tol)
        assert meas.midfoot_width_mm == pytest.approx(expected["midfootWidthMm"], abs=tol)
        assert meas.heel_width_mm == pytest.approx(expected["heelWidthMm"], abs=tol)
        assert meas.heel_to_metatarsal_line_mm == pytest.approx(
            expected["heelToMetatarsalLineMm"], abs=tol)
        assert meas.arch_index == pytest.approx(expected["archIndex"], abs=1e-9)
        assert meas.plantar_area_mm2 == pytest.approx(expected["plantarAreaMm2"], abs=1e-4)
        assert frame.length_mm == pytest.approx(expected["frameLengthMm"], abs=tol)


def test_parity_fixture_has_realistic_cases(parity_fixture):
    assert len(parity_fixture["cases"]) >= 4
    for case in parity_fixture["cases"]:
        assert len(case["input"]["contourMm"]) >= 30
        assert 150.0 < case["expected"]["lengthMm"] < 320.0


# ------------------------------------------------------------------- quality gate


def analyze_scene(**kwargs):
    scene = G.render_scene(G.bilateral_scene(258.0, 261.0, **kwargs))
    decoded = DecodedImage(scene.image_bgr, 1, scene.image_bgr.shape[1],
                           scene.image_bgr.shape[0])
    return analyze(decoded)


def test_good_capture_scores_high():
    result = analyze_scene()
    assert result.quality.passed
    assert result.quality.score() >= 85.0
    assert "excelente" in result.quality.summary().lower()


def test_defocused_capture_is_blocked():
    with pytest.raises(PipelineBlocked) as exc:
        analyze_scene(defocus_px=6)
    report = exc.value.report
    assert not report.passed
    assert any("desfocada" in b.lower() for b in report.blockers)
    # Um bloqueio nunca pode devolver um score que sugira "quase bom".
    assert report.score() <= 49.0


def test_quality_checks_cover_required_dimensions():
    result = analyze_scene()
    ids = {c.id for c in result.quality.checks}
    required = {
        "marker_detected", "marker_corners", "marker_not_clipped", "marker_skew",
        "resolution_px_per_mm", "megapixels", "focus", "motion_blur", "exposure",
        "glare", "perspective", "feet_detected", "feet_complete", "border_margin",
        "foot_bg_contrast", "scale_round_trip",
    }
    assert required <= ids, required - ids


def test_glare_and_motion_are_reported_as_warnings():
    result = analyze_scene(glare=0.5, motion_blur_px=9)
    checks = {c.id: c for c in result.quality.checks}
    assert checks["glare"].score < 0.9
    assert checks["motion_blur"].score < 0.9
    assert result.quality.passed, "avisos não devem bloquear sozinhos"


def test_thresholds_are_configurable(monkeypatch):
    from app import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("PEDIGRAFIA_MIN_FOCUS_SCORE", "5")
    try:
        # O limiar vem do dataclass; garantimos que a leitura de env é o caminho único.
        assert config.get_settings().quality.min_focus_score == 20.0
    finally:
        monkeypatch.delenv("PEDIGRAFIA_MIN_FOCUS_SCORE", raising=False)
        config.get_settings.cache_clear()


def test_segmenter_is_swappable():
    from app.segmentation.factory import available, get_segmenter

    assert {"classical", "onnx"} <= set(available())
    # Modelo ONNX ausente deve cair no clássico, com registro — nunca falhar.
    assert get_segmenter("onnx").name == "classical"
    assert get_segmenter("inexistente").name == "classical"
