"""Calibração por objeto de dimensão normalizada — sem imprimir marcador.

O alvo ArUco continua sendo o método mais exato. Este caminho existe porque a
fricção de imprimir e colar folhas é real, e qualquer retângulo padronizado já traz
para dentro da foto a única coisa que uma fotografia não tem: uma medida.

O que estes testes travam:

* as dimensões do catálogo são as das normas, e a tolerância de cada norma é
  reportada como incerteza de escala — é erro que nenhum algoritmo remove;
* a razão largura/altura do retângulo é recuperável da perspectiva e é usada só
  para **conferir**, nunca para medir;
* o modo automático se **recusa** a escolher quando a forma não prova de qual
  objeto se trata (confundir cartão com A4 erraria a escala em 2,45×);
* um segundo objeto do outro lado dos pés converte extrapolação em interpolação —
  a mesma correção que o alvo de quatro marcadores faz, sem impressora.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.calibration.fit import extrapolation_distance_mm, fit_free_rectangles
from app.calibration.reference import (CATALOGUE, aspect_with_uncertainty,
                                       detect_reference, parse_custom_reference,
                                       rectangle_aspect, reference_target,
                                       resolve_reference)
from app.calibration.target import MarkerPlacement
from app.geometry import polygon as poly
from app.image_io import DecodedImage
from app.pipeline import analyze
from app.synth import generator as G


# --------------------------------------------------------------------------- catálogo


def test_catalogue_dimensions_are_the_standard_ones():
    card = CATALOGUE["card"]
    assert (card.width_mm, card.height_mm) == (85.60, 53.98)   # ISO/IEC 7810 ID-1
    assert card.corner_radius_mm == pytest.approx(3.18)
    a4 = CATALOGUE["a4"]
    assert (a4.width_mm, a4.height_mm) == (210.0, 297.0)       # ISO 216


def test_object_tolerance_becomes_reported_scale_uncertainty():
    """A tolerância do padrão físico é um piso de erro, não um detalhe."""
    card = CATALOGUE["card"]
    a4 = CATALOGUE["a4"]
    # Cartão: ±0,13 mm em 53,98 mm ≈ 0,24 % → ±0,64 mm em um pé de 265 mm.
    assert card.scale_tolerance_rel == pytest.approx(0.13 / 53.98, rel=1e-9)
    assert 265.0 * card.scale_tolerance_rel < 0.7
    # Folha A4: ±2 mm em 210 mm ≈ 0,95 % → ±2,5 mm no mesmo pé. Uma ordem de
    # grandeza pior que o próprio erro geométrico do sistema.
    assert 265.0 * a4.scale_tolerance_rel > 2.0
    assert a4.scale_tolerance_rel > card.scale_tolerance_rel * 3.0


def test_custom_reference_parsing_and_limits():
    ref = parse_custom_reference("85,6x53.98@0.1")
    assert ref.width_mm == pytest.approx(85.6)
    assert ref.tolerance_mm == pytest.approx(0.1)
    with pytest.raises(ValueError):
        parse_custom_reference("85.6")
    with pytest.raises(ValueError):
        parse_custom_reference("2x3")          # fora da faixa plausível
    with pytest.raises(ValueError):
        resolve_reference("moeda")             # não está no catálogo, e não deve estar


def test_rectangular_placement_corners_follow_width_height_and_rotation():
    flat = MarkerPlacement(900, (10.0, 20.0), 85.6, 53.98)
    corners = flat.corners_mm()
    assert np.linalg.norm(corners[1] - corners[0]) == pytest.approx(85.6)
    assert np.linalg.norm(corners[3] - corners[0]) == pytest.approx(53.98)

    turned = MarkerPlacement(900, (10.0, 20.0), 85.6, 53.98, rotation_deg=90.0)
    tc = turned.corners_mm()
    # Girar preserva as dimensões físicas — só muda a orientação no plano.
    assert np.linalg.norm(tc[1] - tc[0]) == pytest.approx(85.6)
    assert tc[1][1] - tc[0][1] == pytest.approx(85.6)

    padded = flat.padded_corners_mm(5.0)
    assert np.linalg.norm(padded[1] - padded[0]) == pytest.approx(85.6 + 10.0)


# ------------------------------------------------------ razão de aspecto projetiva


def _project_rectangle(w_mm, h_mm, focal, tilt, azimuth, roll=13.0, dist=640.0,
                       size=(3000, 4000)):
    """Projeta um retângulo centrado na origem com uma câmera de foco conhecido."""
    w, h = size
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.0]])
    t, a, r = map(math.radians, (tilt, azimuth, roll))
    n = np.array([math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), -math.cos(t)])
    forward = -n
    up = np.array([math.sin(r), -math.cos(r), 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    R = np.stack([right, -np.cross(right, forward), forward], axis=0)
    P = np.stack([R[:, 0], R[:, 1], -R @ (dist * n)], axis=1)
    H = K @ P
    corners = np.array([[-w_mm / 2, -h_mm / 2], [w_mm / 2, -h_mm / 2],
                        [w_mm / 2, h_mm / 2], [-w_mm / 2, h_mm / 2]])
    hom = np.concatenate([corners, np.ones((4, 1))], axis=1) @ H.T
    return hom[:, :2] / hom[:, 2:3], (w, h)


@pytest.mark.parametrize("w_mm,h_mm", [(85.60, 53.98), (210.0, 297.0)])
@pytest.mark.parametrize("azimuth", [37.0, 200.0, 310.0])
def test_projective_aspect_recovers_the_true_ratio(w_mm, h_mm, azimuth):
    """Fora da degenerescência, a razão sai com erro < 1 % e o foco é recuperado."""
    quad, size = _project_rectangle(w_mm, h_mm, 3400.0, 16.0, azimuth)
    ratio, focal = rectangle_aspect(quad, size)
    assert ratio == pytest.approx(w_mm / h_mm, rel=0.01)
    assert focal == pytest.approx(3400.0, rel=0.05)


def test_degenerate_geometry_is_reported_not_guessed():
    """Com o eixo de inclinação paralelo a um lado, um par de lados continua
    paralelo na imagem: não há ponto de fuga e a razão não é observável.

    O módulo precisa **dizer** isso (sigma infinito ou enorme) em vez de devolver
    um número com cara de medida."""
    quad, size = _project_rectangle(85.60, 53.98, 3400.0, 18.0, 0.0, roll=0.0)
    _, sigma, _ = aspect_with_uncertainty(quad, size)
    assert sigma > 0.05

    ok_quad, ok_size = _project_rectangle(85.60, 53.98, 3400.0, 18.0, 40.0)
    ratio, ok_sigma, _ = aspect_with_uncertainty(ok_quad, ok_size)
    assert ok_sigma < sigma
    assert ratio == pytest.approx(85.60 / 53.98, rel=0.01)


# ------------------------------------------------------------------------ detecção


@pytest.fixture(scope="module")
def card_scene():
    return G.render_scene(G.reference_scene(
        250.0, 254.0, key="card", count=1, tilt_deg=12.0, azimuth_deg=40.0,
        roll_deg=8.0, camera_distance_mm=700.0, noise_sigma=2.0, seed=4242))


@pytest.fixture(scope="module")
def two_card_scene():
    return G.render_scene(G.reference_scene(
        250.0, 254.0, key="card", count=2, tilt_deg=12.0, azimuth_deg=40.0,
        roll_deg=8.0, camera_distance_mm=700.0, noise_sigma=2.0, seed=4242))


@pytest.fixture(scope="module")
def four_card_scene():
    return G.render_scene(G.reference_scene(
        250.0, 254.0, key="card", count=4, tilt_deg=12.0, azimuth_deg=40.0,
        roll_deg=8.0, camera_distance_mm=700.0, noise_sigma=2.0, seed=4242))


def test_detects_a_declared_card_with_subpixel_edges(card_scene):
    det = detect_reference(card_scene.image_bgr, declared="card")
    assert det.found
    assert det.reference.key == "card"
    best = det.candidates[0]
    # O cartão do ID-1 tem canto arredondado de 3,18 mm: só o ajuste de retas nas
    # arestas recupera o vértice ideal, e ele tem de ficar sub-pixel.
    assert best.edge_rms_px < 0.6
    assert best.aspect_error_rel < 0.02


def test_auto_mode_refuses_instead_of_guessing_the_object(card_scene):
    """A forma não prova qual objeto é — a série ISO A inteira tem razão √2, e um
    cartão fica a ~11 % dela. Trocar um pelo outro erraria a escala em 2,45×."""
    det = detect_reference(card_scene.image_bgr)
    assert not det.found
    assert det.ambiguous
    assert "Informe qual" in det.detection.reason


def test_rejects_a_photo_without_any_reference():
    plain = np.full((900, 1200, 3), 235, dtype=np.uint8)
    det = detect_reference(plain, declared="card")
    assert not det.found
    assert det.detection.reason


# -------------------------------------------------------- ajuste e cobertura


def test_single_reference_is_exact_and_extrapolates(card_scene):
    det = detect_reference(card_scene.image_bgr, declared="card")
    fit, poses = fit_free_rectangles(det.detection, det.reference.model_mm)
    assert fit.marker_count == 1
    # Quatro pontos determinam a homografia: resíduo nulo por construção, portanto
    # o resíduo NÃO é sinal de qualidade aqui — e o código diz isso.
    assert fit.exact
    # 1e-4 mm é o ruído de `getPerspectiveTransform`, que trabalha em float32; o
    # ponto do teste é que o resíduo é zero geométrico, e portanto não mede nada.
    assert fit.residual_max_mm < 1e-4
    assert poses == [(0.0, 0.0, 0.0)]
    assert any("extrapolada" in w for w in fit.warnings)


def test_second_object_makes_the_fit_overdetermined(two_card_scene):
    det = detect_reference(two_card_scene.image_bgr, declared="card")
    assert len(det.candidates) == 2
    fit, poses = fit_free_rectangles(det.detection, det.reference.model_mm)
    assert fit.marker_count == 2
    assert not fit.exact
    # Agora o resíduo é informação real: os dois cartões precisam ser consistentes
    # com uma única homografia e com a própria forma declarada.
    assert fit.residual_max_mm < 1.0
    # O segundo cartão foi localizado a uma distância física plausível do primeiro.
    assert 120.0 < math.hypot(poses[1][0], poses[1][1]) < 600.0
    span_w, span_h = fit.coverage_span_mm()
    assert max(span_w, span_h) > 200.0


def test_reference_target_describes_what_produced_the_scale():
    ref = resolve_reference("card")
    target = reference_target(ref, [(0.0, 0.0, 0.0), (240.0, 30.0, -7.0)])
    assert target.kind == "reference"
    assert len(target.markers) == 2
    assert target.scale_tolerance_rel() == pytest.approx(ref.scale_tolerance_rel)
    assert "ISO/IEC 7810 ID-1" in target.description


# ------------------------------------------------------------- pipeline completo


def _length_errors(scene, calibration):
    """Erro de comprimento por pé, emparelhado por **lateralidade**.

    Emparelhar por posição não serve aqui: o referencial em mm é ancorado no objeto
    de referência, e com vários objetos ele pode estar girado em relação ao plano da
    cena. Comprimento (maior corda do casco) e lateralidade (geometria do hálux) são
    ambos invariantes a isso; posição não é.
    """
    result = analyze(DecodedImage(scene.image_bgr, 1, scene.image_bgr.shape[1],
                                  scene.image_bgr.shape[0]),
                     calibration=calibration)
    truth_by_side = dict(zip(scene.truth_lateralities, scene.truth_contours_mm))
    errors = []
    for foot in result.feet:
        truth = truth_by_side.get(foot.laterality)
        assert truth is not None, f"lateralidade inesperada: {foot.laterality}"
        truth_len, _, _ = poly.max_caliper(truth)
        got_len, _, _ = poly.max_caliper(foot.contour_high_res_mm)
        errors.append(got_len - truth_len)
    return errors, result


def test_card_calibrated_pipeline_measures_in_real_millimetres(card_scene):
    errors, result = _length_errors(card_scene, "card")
    assert len(errors) == 2
    assert result.marker.source == "reference"
    assert result.target.kind == "reference"
    # Um cartão é pequeno e fica longe dos pés: a escala é extrapolada e o erro é
    # maior que o do alvo de quatro marcadores. O que se exige aqui é que continue
    # sendo uma medida física, não que seja tão boa quanto o alvo impresso.
    assert max(abs(e) for e in errors) < 3.0
    assert result.extrapolation_mm > 0.0


def test_more_objects_shrink_the_extrapolated_region(card_scene, two_card_scene,
                                                     four_card_scene):
    """Dois cartões cobrem uma faixa; quatro cobrem uma área.

    O casco convexo dos pontos de controle é o que decide entre interpolar e
    extrapolar, e dois objetos pequenos definem quase uma reta — só fecham a área
    quando são quatro. É a mesma geometria do alvo impresso, sem impressora.

    O alvo não é extrapolação zero: nem o tabuleiro impresso consegue isso, porque
    os pododáctilos e o calcâneo passam um pouco além dos marcadores (medido: 21 mm
    em média para o `board4` nas mesmas condições). O que se exige aqui é chegar a
    esse mesmo patamar — bem abaixo do limiar de aviso — partindo de ~290 mm com um
    cartão só.
    """
    from app.config import get_settings

    _, one = _length_errors(card_scene, "card")
    _, two = _length_errors(two_card_scene, "card")
    errors, four = _length_errors(four_card_scene, "card")
    assert (one.calibration.marker_count, two.calibration.marker_count,
            four.calibration.marker_count) == (1, 2, 4)
    assert four.extrapolation_mm < two.extrapolation_mm < one.extrapolation_mm
    assert four.extrapolation_mm < 0.15 * one.extrapolation_mm
    assert four.extrapolation_mm < get_settings().max_extrapolation_warn_mm
    assert max(abs(e) for e in errors) < 1.0


def test_round_trip_re_measures_the_object_on_the_rectified_image(card_scene):
    _, result = _length_errors(card_scene, "card")
    assert result.round_trip_sides_mm, "a referência precisa ser re-medida"
    assert result.round_trip_error_mm < 0.5
    # Lados e diagonais: 4 + 2 valores.
    assert len(result.round_trip_sides_mm) == 6


def test_pipeline_reports_the_irreducible_scale_uncertainty(card_scene):
    _, result = _length_errors(card_scene, "card")
    text = " ".join(result.warnings)
    assert "tolerância dimensional" in text.lower()
    assert "nenhum processamento" in text.lower()


def test_aruco_mode_never_falls_back_to_a_reference(card_scene):
    """Quem pediu marcador impresso não pode receber, em silêncio, uma escala
    derivada de outra coisa."""
    from app.pipeline import PipelineBlocked

    with pytest.raises(PipelineBlocked):
        analyze(DecodedImage(card_scene.image_bgr, 1, card_scene.image_bgr.shape[1],
                             card_scene.image_bgr.shape[0]), calibration="aruco")


def test_feet_are_not_confused_with_the_reference_object(card_scene):
    _, result = _length_errors(card_scene, "card")
    assert len(result.feet) == 2
    for foot in result.feet:
        length, _, _ = poly.max_caliper(foot.contour_high_res_mm)
        assert 200.0 < length < 320.0


def test_declaring_the_wrong_object_is_refused(card_scene):
    """Falha mais perigosa deste caminho: declarar o objeto errado.

    Duas barreiras, nesta ordem — o filtro de razão de aspecto, quando as formas
    diferem o bastante (cartão 1,586 contra A4 1,414: 12 %), e o limite físico de
    comprimento plantar (90–400 mm), que reprova a captura em vez de entregar um
    molde coerente e proporcionalmente errado.

    Nenhuma das duas cobre tudo: ver `test_iso_a_series_hazard_is_declared`.
    """
    from app.pipeline import PipelineBlocked

    decoded = DecodedImage(card_scene.image_bgr, 1, card_scene.image_bgr.shape[1],
                           card_scene.image_bgr.shape[0])
    with pytest.raises(PipelineBlocked):
        analyze(decoded, calibration="a4")


def test_iso_a_series_hazard_is_declared(card_scene, monkeypatch):
    """O buraco que nenhuma verificação fecha, dito em voz alta.

    A4, A5 e A3 têm a mesma proporção √2, então a foto não distingue uma da outra.
    Declarar A4 para uma folha A5 multiplica tudo por 1,42 — e um pé de 265 mm
    viraria 376 mm, ainda dentro do limite físico de 400 mm. Não há como o sistema
    pegar isso; o que ele pode fazer é avisar, e é isso que este teste trava.
    """
    scene = G.render_scene(G.reference_scene(
        250.0, 254.0, key="a4", count=1, tilt_deg=10.0, azimuth_deg=35.0,
        roll_deg=6.0, camera_distance_mm=900.0, noise_sigma=2.0, seed=99))
    _, result = _length_errors(scene, "a4")
    text = " ".join(result.warnings)
    assert "não é verificável" in text.lower() or "NÃO é verificável" in text
    assert "√2" in text
    assert "42 %" in text


# ------------------------------------------------------------------------- API


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app
    from app.storage import store

    with TestClient(app) as c:
        yield c
    store.purge_all()


def test_catalogue_endpoint_publishes_the_error_floor(client):
    body = client.get("/api/references").json()
    keys = {o["key"] for o in body["objects"]}
    assert {"card", "a4"} <= keys
    a4 = next(o for o in body["objects"] if o["key"] == "a4")
    assert a4["scaleToleranceMmAt265"] > 2.0
    assert a4["standard"] == "ISO 216"


def test_coin_and_ruler_are_refused_with_the_arithmetic(client):
    """Foram pedidos e não entraram. O motivo é numérico e fica publicado."""
    rejected = {o["key"]: o["reason"] for o in client.get("/api/references").json()["rejected"]}
    assert set(rejected) == {"coin", "ruler"}
    assert "elipse" in rejected["coin"]
    assert "3,4 mm" in rejected["coin"]
    assert "colineares" in rejected["ruler"]


def test_analyze_rejects_an_unknown_calibration_source(client, card_scene):
    res = client.post("/api/analyze",
                      files={"image": ("c.png", card_scene.encode_png(), "image/png")},
                      data={"calibration": "moeda"})
    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "bad_calibration"


def test_analyze_reports_the_reference_it_used(client, card_scene):
    res = client.post("/api/analyze",
                      files={"image": ("c.png", card_scene.encode_png(), "image/png")},
                      data={"calibration": "card"})
    assert res.status_code == 200, res.text
    calib = res.json()["calibration"]
    assert calib["source"] == "reference"
    assert calib["reference"]["key"] == "card"
    assert calib["reference"]["standard"] == "ISO/IEC 7810 ID-1"
    assert calib["scaleToleranceMm"] > 0.0
    assert len(calib["controlPointsMm"]) == 4
