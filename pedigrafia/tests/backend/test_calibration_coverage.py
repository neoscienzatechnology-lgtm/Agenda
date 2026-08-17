"""Calibração distribuída: o alvo de quatro marcadores elimina a extrapolação.

Estes testes travam a maior correção de exatidão do sistema. A causa do viés de até
2 mm foi **medida** (ver ``docs/METROLOGY_CALIBRATION.md``): com um marcador só, a
homografia é exata sobre ele e extrapola para o resto da plataforma; o erro aparece
sempre no extremo do pé mais distante do marcador e cresce com a inclinação da câmera.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.calibration.fit import extrapolation_distance_mm, fit_homography
from app.calibration.marker import detect_marker
from app.calibration.target import (board4_target, load_target_from_json,
                                    select_target_for_detection,
                                    single_marker_target)
from app.geometry import polygon as poly
from app.image_io import DecodedImage
from app.pipeline import analyze
from app.synth import generator as G

TILTS = [(0.0, 0.0), (10.0, 40.0), (18.0, 150.0), (25.0, 260.0)]


def measure(scene) -> list[float]:
    result = analyze(DecodedImage(scene.image_bgr, 1, scene.image_bgr.shape[1],
                                  scene.image_bgr.shape[0]))
    feet = sorted(result.feet, key=lambda f: float(f.component.centroid_mm[0]))
    errors = []
    for foot, truth in zip(feet, scene.truth_contours_mm):
        truth_len, _, _ = poly.max_caliper(truth)
        got_len, _, _ = poly.max_caliper(foot.contour_high_res_mm)
        errors.append(got_len - truth_len)
    return errors, result


# ------------------------------------------------------------------ definição do alvo


def test_board_places_four_markers_around_the_area():
    target = board4_target(area_mm=(320.0, 480.0))
    assert len(target.markers) == 4
    assert target.ids == {0, 1, 2, 3}
    span_w, span_h = target.span_mm()
    assert span_w == pytest.approx(320.0)
    assert span_h == pytest.approx(480.0)


def test_auto_selects_board_when_two_or_more_markers_are_visible():
    assert select_target_for_detection([0, 1, 2, 3]).name == "board4"
    assert select_target_for_detection([0, 2]).name == "board4"
    assert select_target_for_detection([7]).name == "single"
    assert select_target_for_detection([]).name == "single"


def test_custom_target_from_json(tmp_path):
    path = tmp_path / "alvo.json"
    path.write_text(
        '{"name":"bancada","markers":['
        '{"id":0,"originMm":[0,0],"sizeMm":50},'
        '{"id":1,"originMm":[300,0],"sizeMm":50}]}', encoding="utf-8")
    target = load_target_from_json(path)
    assert target.name == "bancada"
    assert len(target.markers) == 2
    assert target.placement(1).origin_mm == (300.0, 0.0)


# ---------------------------------------------------------------- ajuste e resíduo


def test_single_marker_fit_is_exact_and_says_so():
    scene = G.render_scene(G.bilateral_scene(258.0, 261.0))
    det = detect_marker(scene.image_bgr)
    fit = fit_homography(det, single_marker_target(marker_id=det.marker_id))
    assert fit.exact
    assert fit.marker_count == 1
    # Com 4 pontos o resíduo é zero por construção: ele NÃO mede qualidade.
    assert fit.residual_max_mm < 1e-6
    assert any("extrapolada" in w for w in fit.warnings)


def test_board_fit_is_overdetermined_and_residual_is_meaningful():
    scene = G.render_scene(G.board_scene(258.0, 261.0, tilt_deg=12.0, azimuth_deg=70.0))
    det = detect_marker(scene.image_bgr)
    fit = fit_homography(det, board4_target())
    assert not fit.exact
    assert fit.marker_count == 4
    assert len(fit.control_points_mm) == 16
    # Agora o resíduo é um sinal real — e precisa ser pequeno.
    assert fit.residual_max_mm < 0.5, fit.residual_max_mm


def test_extrapolation_distance_is_zero_inside_the_board():
    scene = G.render_scene(G.board_scene(258.0, 261.0))
    det = detect_marker(scene.image_bgr)
    fit = fit_homography(det, board4_target())
    inside = np.array([[160.0, 240.0], [100.0, 300.0], [220.0, 180.0]])
    assert extrapolation_distance_mm(fit, inside) == pytest.approx(0.0, abs=1e-6)
    outside = np.array([[160.0, 900.0]])
    assert extrapolation_distance_mm(fit, outside) > 400.0


# ------------------------------------------------------------------- exatidão medida


@pytest.mark.parametrize("tilt,azimuth", TILTS)
def test_board_keeps_error_under_half_millimetre_at_any_tilt(tilt, azimuth):
    """O ganho principal: erro plano com a inclinação, porque não há extrapolação."""
    scene = G.render_scene(G.board_scene(258.0, 261.0, tilt_deg=tilt,
                                         azimuth_deg=azimuth))
    errors, result = measure(scene)
    assert result.target.name == "board4"
    assert result.extrapolation_mm == pytest.approx(0.0, abs=1e-6)
    for err in errors:
        assert abs(err) <= 0.5, f"erro {err:+.2f} mm a {tilt}°"


def test_board_beats_single_marker_under_tilt():
    """Comparação direta, mesma cena geométrica, só mudando o alvo de calibração."""
    board_worst = 0.0
    single_worst = 0.0
    for tilt, azimuth in TILTS:
        b, _ = measure(G.render_scene(
            G.board_scene(258.0, 261.0, tilt_deg=tilt, azimuth_deg=azimuth)))
        s, _ = measure(G.render_scene(
            G.bilateral_scene(258.0, 261.0, tilt_deg=tilt, azimuth_deg=azimuth)))
        board_worst = max(board_worst, max(abs(e) for e in b))
        single_worst = max(single_worst, max(abs(e) for e in s))
    assert board_worst < single_worst, (board_worst, single_worst)
    assert board_worst <= 0.5
    # O marcador único continua funcionando — só é menos exato longe do marcador.
    assert single_worst <= 2.5


def test_round_trip_with_board_checks_distances_across_the_platform():
    """Com vários marcadores a verificação de escala deixa de ter braço de alavanca zero."""
    scene = G.render_scene(G.board_scene(258.0, 261.0, tilt_deg=15.0, azimuth_deg=95.0))
    _, result = measure(scene)
    # 4 marcadores × 4 lados + 6 distâncias entre centros.
    assert len(result.round_trip_sides_mm) == 22
    assert result.round_trip_error_mm < 0.5


# ------------------------------------------------------------------ quality gate


def test_single_marker_warns_about_extrapolation_without_blocking():
    scene = G.render_scene(G.bilateral_scene(258.0, 261.0))
    _, result = measure(scene)
    check = next(c for c in result.quality.checks if c.id == "calibration_coverage")
    assert check.value > 100.0, "os pés estão longe do marcador nesta cena"
    assert check.passed, "extrapolar reduz exatidão mas não deve impedir o uso"
    assert check.score < 0.8, "o score precisa refletir a perda de exatidão"
    assert any("EXTRAPOLADA" in w or "extrapolada" in w for w in result.warnings)


def test_board_reports_interpolated_calibration():
    scene = G.render_scene(G.board_scene(258.0, 261.0))
    _, result = measure(scene)
    check = next(c for c in result.quality.checks if c.id == "calibration_coverage")
    assert check.passed and check.score == pytest.approx(1.0, abs=1e-6)
    residual = next(c for c in result.quality.checks if c.id == "calibration_residual")
    assert residual.passed
    assert not any("extrapolada" in w for w in result.warnings)
