"""Metrologia fim-a-fim: da foto sintética à medida em milímetros."""

from __future__ import annotations

import numpy as np
import pytest

from app.image_io import DecodedImage
from app.pipeline import PipelineBlocked, analyze
from app.synth import generator as G

# Tolerâncias MEDIDAS, não desejadas. A cadeia metrológica (marcador → mm) é exata a
# ~0,07 mm; o erro dominante é a fidelidade da SILHUETA na região dos pododáctilos,
# que depende do borramento efetivo da foto. Com a câmera aproximadamente
# perpendicular o erro fica abaixo de 0,5 mm; com inclinação, os entalhes
# interdigitais deixam de ser resolvidos e a silhueta incha para fora.
# Ver docs/LIMITATIONS.md para os números completos.
TOLERANCE_MM = 1.0          # captura aproximadamente perpendicular
TOLERANCE_TILTED_MM = 2.5   # inclinação de câmera até ~20°


def run(scene) -> object:
    decoded = DecodedImage(scene.image_bgr, 1, scene.image_bgr.shape[1],
                           scene.image_bgr.shape[0])
    return analyze(decoded)


@pytest.mark.parametrize("length", [200.0, 240.0, 260.0, 265.0, 270.0])
def test_single_foot_length_within_one_millimetre(length):
    scene = G.render_scene(G.single_foot_scene(length, "right"))
    result = run(scene)
    assert len(result.feet) == 1
    measured = result.feet[0].measurements.length_mm
    assert measured == pytest.approx(length, abs=TOLERANCE_MM), \
        f"medido {measured:.2f} mm para verdade {length} mm"


def test_bilateral_lengths_and_sides():
    scene = G.render_scene(G.bilateral_scene(258.0, 261.0))
    result = run(scene)
    assert len(result.feet) == 2
    by_side = {f.laterality: f for f in result.feet}
    assert set(by_side) == {"left", "right"}
    assert by_side["right"].measurements.length_mm == pytest.approx(261.0, abs=TOLERANCE_MM)
    assert by_side["left"].measurements.length_mm == pytest.approx(258.0, abs=TOLERANCE_MM)
    for foot in result.feet:
        assert foot.laterality_confidence > 0.5


@pytest.mark.parametrize("tilt,azimuth", [(0.0, 0.0), (10.0, 40.0), (18.0, 150.0)])
def test_length_under_camera_tilt(tilt, azimuth):
    scene = G.render_scene(
        G.single_foot_scene(265.0, "left", tilt_deg=tilt, azimuth_deg=azimuth))
    result = run(scene)
    tol = TOLERANCE_MM if tilt < 5.0 else TOLERANCE_TILTED_MM
    measured = result.feet[0].measurements.length_mm
    assert measured == pytest.approx(265.0, abs=tol), f"{measured:.2f} mm a {tilt}°"
    # O viés é sempre para MAIS (silhueta inflada), nunca para menos: um molde
    # levemente maior é recuperável no acabamento; um menor não é.
    assert measured >= 265.0 - TOLERANCE_MM


def test_length_is_stable_under_capture_conditions():
    """Distância, resolução e compressão diferentes não podem mudar a medida."""
    variants = {
        "base": {},
        "longe": {"camera_distance_mm": 1100.0},
        "perto": {"camera_distance_mm": 480.0},
        "baixa_res": {"image_size": (1200, 1560)},
        "jpeg": {"jpeg_quality": 70},
        "ruido": {"noise_sigma": 6.0},
    }
    measured = {}
    for name, kwargs in variants.items():
        scene = G.render_scene(G.single_foot_scene(265.0, "right", **kwargs))
        try:
            result = run(scene)
        except PipelineBlocked as exc:
            pytest.fail(f"variante '{name}' reprovada: {exc.report.blockers}")
        measured[name] = result.feet[0].measurements.length_mm

    for name, value in measured.items():
        assert value == pytest.approx(265.0, abs=TOLERANCE_MM), f"{name}: {value:.2f} mm"
    spread = max(measured.values()) - min(measured.values())
    assert spread < 0.6, f"dispersão entre capturas: {spread:.2f} mm ({measured})"


def test_camera_roll_degrades_toes_but_stays_bounded():
    """Rolagem da câmera borra os entalhes interdigitais e infla a silhueta.

    Documentado como limitação conhecida: o erro é monotônico com a rolagem e
    permanece dentro da tolerância de captura inclinada.
    """
    errors = {}
    for roll in (0.0, 8.0, 17.0):
        scene = G.render_scene(G.single_foot_scene(265.0, "right", roll_deg=roll))
        errors[roll] = run(scene).feet[0].measurements.length_mm - 265.0
    assert abs(errors[0.0]) < 0.5
    assert all(e <= TOLERANCE_TILTED_MM for e in errors.values()), errors
    assert errors[17.0] >= errors[0.0], "o viés deve ser para mais, nunca para menos"


def test_round_trip_check_reported_in_result():
    scene = G.render_scene(G.single_foot_scene(265.0))
    result = run(scene)
    assert result.round_trip_error_mm < 0.3
    assert len(result.round_trip_sides_mm) == 4
    for side in result.round_trip_sides_mm:
        assert side == pytest.approx(50.0, abs=0.3)


def test_widths_are_plausible_and_ordered():
    scene = G.render_scene(G.single_foot_scene(265.0))
    m = run(scene).feet[0].measurements
    assert m.forefoot_width_mm > m.heel_width_mm > m.midfoot_width_mm
    assert 90.0 < m.forefoot_width_mm < 120.0
    assert 0 < m.heel_to_metatarsal_line_mm < m.length_mm


def test_dark_background_podoscope_still_measures():
    scene = G.render_scene(
        G.single_foot_scene(265.0, "left", background="dark", skin_bgr=(120, 145, 178)))
    result = run(scene)
    assert result.feet[0].measurements.length_mm == pytest.approx(265.0, abs=TOLERANCE_MM)


def test_all_geometry_is_expressed_in_millimetres():
    """Nenhum contorno pode sair em coordenadas de pixel."""
    scene = G.render_scene(G.single_foot_scene(265.0))
    result = run(scene)
    foot = result.feet[0]
    truth = scene.truth_contours_mm[0]
    lo_t, hi_t = truth.min(axis=0), truth.max(axis=0)
    lo_m, hi_m = foot.contour_high_res_mm.min(axis=0), foot.contour_high_res_mm.max(axis=0)
    assert np.allclose(lo_m, lo_t, atol=2.0)
    assert np.allclose(hi_m, hi_t, atol=2.0)


def test_landmarks_are_reported_in_foot_frame():
    from app.pipeline import foot_local_coords

    scene = G.render_scene(G.single_foot_scene(265.0))
    foot = run(scene).feet[0]
    coords = {c["id"]: c for c in foot_local_coords(foot)}
    for key in ("T1", "T2", "T3", "T4", "T5", "M1", "M2", "M3", "M4", "M5", "H"):
        assert key in coords, f"{key} ausente no frame do pé"
    assert coords["T1"]["tPct"] > 90.0
    assert coords["H"]["tPct"] < 25.0
    assert 50.0 < coords["M1"]["tPct"] < 90.0
