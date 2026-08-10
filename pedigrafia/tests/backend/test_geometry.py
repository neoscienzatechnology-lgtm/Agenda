"""Contorno, simplificação, lateralidade e medidas sobre geometria exata."""

from __future__ import annotations

import numpy as np
import pytest

from app.geometry import polygon as poly
from app.geometry.contour import prepare_contour, simplify_to_target
from app.geometry.frame import bootstrap_frame, refine_frame
from app.geometry.laterality import classify
from app.geometry.shape import shape_stats
from app.landmarks.metatarsals import estimate_metatarsal_heads
from app.landmarks.toes import detect_toe_apices
from app.measurements.compute import compute_measurements, shoe_size_check
from app.synth import foot_shape as FS


def ideal(length_mm=265.0, laterality="right", view="below", rotation=0.0):
    contour = FS.foot_polygon(length_mm, laterality, view, rotation_deg=rotation)
    lm = FS.expected_landmarks_mm(length_mm, laterality, view, rotation_deg=rotation)
    frame, _ = bootstrap_frame(contour)
    frame = refine_frame(contour, frame, lm["T2"])
    return contour, frame, lm


# ------------------------------------------------------------------- primitivas


def test_polygon_area_of_unit_square():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert poly.area(square) == pytest.approx(100.0)
    assert poly.perimeter(square) == pytest.approx(40.0)
    assert poly.centroid(square) == pytest.approx([5.0, 5.0])


def test_clip_band_preserves_expected_area():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    band = poly.clip_band_u(square, 2.0, 5.0)
    assert poly.area(band) == pytest.approx(30.0)


def test_width_at_u_on_square():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert poly.width_at_u(square, 5.0) == pytest.approx(10.0)


# ---------------------------------------------------------------------- contorno


@pytest.mark.parametrize("length", [200.0, 240.0, 265.0, 290.0])
def test_smoothing_preserves_length(length):
    """I3: suavizar não pode custar milímetros reais."""
    contour, _, _ = ideal(length)
    pair = prepare_contour(contour)
    assert pair.smoothing_drift_mm <= 0.5


@pytest.mark.parametrize("length", [200.0, 265.0])
def test_simplify_preserves_geometry_and_node_budget(length):
    """I4: os nós editáveis mantêm o erro geométrico baixo."""
    contour, _, _ = ideal(length)
    pair = prepare_contour(contour)
    assert pair.simplify_error_mm <= 0.30
    assert 40 <= pair.node_count <= 160
    dense_len = _axis_length(pair.high_res_mm)
    simple_len = _axis_length(pair.editable_mm)
    assert abs(dense_len - simple_len) <= 0.3


def test_simplify_to_target_respects_error_budget():
    contour, _, _ = ideal(265.0)
    dense = poly.resample_closed(contour, 0.5)
    simplified, err = simplify_to_target(dense, 96, 0.3, 40, 160)
    assert err <= 0.3
    assert len(simplified) <= 160


def _axis_length(contour):
    u = poly.principal_axis(contour)
    proj = np.asarray(contour) @ u
    return float(np.max(proj) - np.min(proj))


# ------------------------------------------------------------------ lateralidade


@pytest.mark.parametrize("laterality", ["left", "right"])
@pytest.mark.parametrize("view", ["below", "above"])
@pytest.mark.parametrize("rotation", [0.0, 9.0, -14.0])
def test_laterality_from_geometry(laterality, view, rotation):
    """D/E vem da anatomia, nunca da posição na imagem."""
    contour, frame, _ = ideal(265.0, laterality, view, rotation)
    toes, hallux_sign, hallux_conf = detect_toe_apices(contour, frame)
    assert len(toes) == 5
    result = classify(contour, frame, hallux_sign, hallux_conf, view)
    assert result.laterality == laterality
    assert result.confidence > 0.5


def test_laterality_is_independent_of_translation():
    """Mover o pé na imagem não pode mudar a classificação."""
    results = []
    for center in ((-300.0, -300.0), (0.0, 0.0), (400.0, 250.0)):
        contour = FS.foot_polygon(265.0, "right", "below", center_mm=center)
        lm = FS.expected_landmarks_mm(265.0, "right", "below", center_mm=center)
        frame, _ = bootstrap_frame(contour)
        frame = refine_frame(contour, frame, lm["T2"])
        toes, sign, conf = detect_toe_apices(contour, frame)
        results.append(classify(contour, frame, sign, conf, "below").laterality)
    assert set(results) == {"right"}


# --------------------------------------------------------------------- landmarks


def test_toe_apices_match_model_within_tolerance():
    contour, frame, lm = ideal(265.0)
    toes, _, _ = detect_toe_apices(contour, frame)
    assert len(toes) == 5
    for i, toe in enumerate(toes):
        expected = lm[f"T{i + 1}"]
        assert float(np.linalg.norm(toe.point_mm - expected)) < 9.0, f"T{i+1}"


def test_metatarsal_extremes_are_observed_and_middles_flagged_estimated():
    contour, frame, lm = ideal(265.0)
    _, sign, _ = detect_toe_apices(contour, frame)
    heads, conf = estimate_metatarsal_heads(contour, frame, sign)
    assert len(heads) == 5
    assert heads[0].method.startswith("contour_extreme")
    assert heads[4].method.startswith("contour_extreme")
    for i in (1, 2, 3):
        assert heads[i].method == "interpolated_model"
        assert heads[i].confidence <= 0.5, "M2–M4 não podem parecer observados"
    assert float(np.linalg.norm(heads[0].point_mm - lm["M1"])) < 12.0
    assert float(np.linalg.norm(heads[4].point_mm - lm["M5"])) < 12.0
    assert 0 < conf <= 1


# ----------------------------------------------------------------------- medidas


@pytest.mark.parametrize("length", [200.0, 240.0, 260.0, 265.0, 270.0, 290.0])
def test_length_from_ideal_contour_is_exact(length):
    contour, frame, lm = ideal(length)
    meas, _ = compute_measurements(contour, frame, lm["M1"], lm["M5"], 0.78)
    assert meas.length_mm == pytest.approx(length, abs=0.05)


def test_measures_scale_linearly_with_size():
    small, frame_s, lm_s = ideal(200.0)
    big, frame_b, lm_b = ideal(300.0)
    ms, _ = compute_measurements(small, frame_s, lm_s["M1"], lm_s["M5"], 0.78)
    mb, _ = compute_measurements(big, frame_b, lm_b["M1"], lm_b["M5"], 0.78)
    ratio = 300.0 / 200.0
    assert mb.forefoot_width_mm / ms.forefoot_width_mm == pytest.approx(ratio, rel=0.01)
    assert mb.heel_width_mm / ms.heel_width_mm == pytest.approx(ratio, rel=0.01)
    assert mb.plantar_area_mm2 / ms.plantar_area_mm2 == pytest.approx(ratio ** 2, rel=0.02)
    # Índice de arco é adimensional: não pode depender do tamanho.
    assert mb.arch_index == pytest.approx(ms.arch_index, abs=0.01)


def test_measures_are_invariant_to_rotation():
    base, frame_b, lm_b = ideal(265.0, rotation=0.0)
    rot, frame_r, lm_r = ideal(265.0, rotation=31.0)
    mb, _ = compute_measurements(base, frame_b, lm_b["M1"], lm_b["M5"], 0.78)
    mr, _ = compute_measurements(rot, frame_r, lm_r["M1"], lm_r["M5"], 0.78)
    assert mr.length_mm == pytest.approx(mb.length_mm, abs=0.05)
    assert mr.forefoot_width_mm == pytest.approx(mb.forefoot_width_mm, abs=0.1)
    assert mr.arch_index == pytest.approx(mb.arch_index, abs=0.005)


def test_arch_index_is_labelled_silhouette_with_warning():
    contour, frame, lm = ideal(265.0)
    meas, arch = compute_measurements(contour, frame, lm["M1"], lm["M5"], 0.78)
    assert meas.arch_index_basis == "silhouette"
    assert "SILHUETA" in arch.warning
    assert "contato" in arch.warning


def test_shape_stats_reject_non_foot_blob():
    strip = np.array([[0, 0], [300, 0], [300, 30], [0, 30]], float)
    assert shape_stats(strip).likeness < 0.12
    contour, _, _ = ideal(265.0)
    assert shape_stats(contour).likeness > 0.4


# ---------------------------------------------------------- número do calçado


def test_shoe_size_is_advisory_only():
    """I10: o número informado nunca altera dimensão alguma."""
    check = shoe_size_check(264.3, 40, "BR")
    assert check["measuredMm"] == 264.3
    assert check["compatible"] is True
    assert "compatível" in check["message"]

    off = shoe_size_check(230.0, 44, "BR")
    assert off["compatible"] is False
    assert off["measuredMm"] == 230.0
    assert "medido pela calibração" in off["message"]


def test_shoe_size_absent_produces_no_claim():
    assert shoe_size_check(264.3, None)["message"] == ""
