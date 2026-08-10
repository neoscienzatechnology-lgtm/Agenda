"""O teste que dá sentido ao produto: o PDF impresso tem a dimensão física medida."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import A4_HEIGHT_PT, A4_WIDTH_PT, MM_TO_PT, PT_TO_MM
from app.geometry.frame import bootstrap_frame, refine_frame
from app.pdf.builder import CONTOUR_RGB, build_foot_pdf, build_marker_sheet_pdf
from app.pdf.geometry_page import compute_placement, is_isometry
from app.pdf.inspect import inspect_pdf, path_axis_length_mm, path_max_caliper_mm
from app.synth import foot_shape as FS

LENGTHS = [200.0, 240.0, 260.0, 265.0, 270.0]


def make_pdf(length_mm: float, laterality: str = "right", view: str = "below"):
    contour = FS.foot_polygon(length_mm, laterality, view, center_mm=(37.0, -90.0))
    lm = FS.expected_landmarks_mm(length_mm, laterality, view, center_mm=(37.0, -90.0))
    frame, _ = bootstrap_frame(contour)
    frame = refine_frame(contour, frame, lm["T2"])
    uv = frame.to_local(contour)
    a = frame.to_plane(np.array([[float(np.min(uv[:, 0])), 0.0]]))[0]
    b = frame.to_plane(np.array([[float(np.max(uv[:, 0])), 0.0]]))[0]
    result = build_foot_pdf(
        contour_mm=contour, axis_a_mm=a, axis_b_mm=b, laterality=laterality,
        measurements={
            "lengthMm": length_mm, "forefootWidthMm": 100.0, "midfootWidthMm": 70.0,
            "heelWidthMm": 70.0, "heelToMetatarsalLineMm": 170.0, "archIndex": 0.25,
        },
        landmarks={k: v.tolist() for k, v in lm.items() if k.startswith("M")},
        view=view,
    )
    return result, contour, (a, b)


def test_mm_to_pt_constant_is_exact():
    assert MM_TO_PT == pytest.approx(72.0 / 25.4, abs=0.0)
    assert MM_TO_PT * PT_TO_MM == pytest.approx(1.0, abs=1e-15)
    assert 25.4 * MM_TO_PT == pytest.approx(72.0, abs=1e-12)


@pytest.mark.parametrize("length", LENGTHS)
def test_mediabox_is_exactly_a4(length):
    """I5: a página tem 210 × 297 mm exatos."""
    result, _, _ = make_pdf(length)
    info = inspect_pdf(result.data)
    assert info.is_a4()
    x0, y0, x1, y1 = info.media_box_pt
    assert (x1 - x0) == pytest.approx(A4_WIDTH_PT, abs=0.001)
    assert (y1 - y0) == pytest.approx(A4_HEIGHT_PT, abs=0.001)
    w_mm, h_mm = info.media_box_mm
    assert w_mm == pytest.approx(210.0, abs=0.001)
    assert h_mm == pytest.approx(297.0, abs=0.001)


@pytest.mark.parametrize("length", LENGTHS)
def test_contour_length_survives_the_pdf(length):
    """I6: o comprimento relido do arquivo é o comprimento medido."""
    result, _, _ = make_pdf(length)
    info = inspect_pdf(result.data)
    paths = info.paths_with_color(CONTOUR_RGB)
    assert len(paths) == 1, "o contorno deve ser um único path na cor reservada"
    measured = path_max_caliper_mm(paths[0])
    assert measured == pytest.approx(length, abs=0.2), f"lido {measured:.4f} mm"


@pytest.mark.parametrize("length", LENGTHS)
def test_pdf_error_is_far_below_one_millimetre(length):
    result, _, _ = make_pdf(length)
    info = inspect_pdf(result.data)
    measured = path_max_caliper_mm(info.paths_with_color(CONTOUR_RGB)[0])
    assert abs(measured - length) <= 1.0
    assert abs(measured - length) / length <= 0.005     # meta de <0,5 %


def test_placement_is_rigid():
    """I7: o posicionamento preserva TODAS as distâncias par-a-par."""
    result, contour, _ = make_pdf(265.0)
    assert is_isometry(result.placement.matrix)
    assert abs(abs(float(np.linalg.det(result.placement.matrix))) - 1.0) < 1e-12

    page = result.placement.to_page_mm(contour)
    idx = np.arange(0, len(contour), 5)
    d_plane = np.linalg.norm(contour[idx][:, None, :] - contour[idx][None, :, :], axis=2)
    d_page = np.linalg.norm(page[idx][:, None, :] - page[idx][None, :, :], axis=2)
    assert float(np.max(np.abs(d_plane - d_page))) < 1e-9


def test_no_scaling_operator_in_content_stream():
    """Uma matriz `cm` com escala ≠ 1 quebraria o 1:1 silenciosamente."""
    import re

    from pypdf import PdfReader
    import io

    result, _, _ = make_pdf(265.0)
    data = PdfReader(io.BytesIO(result.data)).pages[0].get_contents().get_data()
    for m in re.finditer(rb"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+"
                         rb"([-\d.]+)\s+([-\d.]+)\s+cm", data):
        a, b, c, d = (float(m.group(i)) for i in (1, 2, 3, 4))
        sx = (a * a + b * b) ** 0.5
        sy = (c * c + d * d) ** 0.5
        assert sx == pytest.approx(1.0, abs=1e-9), "escala horizontal no content stream"
        assert sy == pytest.approx(1.0, abs=1e-9), "escala vertical no content stream"


def test_metadata_declares_scale_and_length():
    result, _, _ = make_pdf(265.0)
    info = inspect_pdf(result.data)
    keywords = info.metadata.get("/Keywords", "")
    assert "scale=1:1" in keywords
    assert "lengthMm=265.00" in keywords
    assert "unit=mm" in keywords


def test_one_foot_per_page():
    from pypdf import PdfReader
    import io

    result, _, _ = make_pdf(265.0)
    assert len(PdfReader(io.BytesIO(result.data)).pages) == 1


@pytest.mark.parametrize("view,expected_mirror", [("below", True), ("above", False)])
def test_view_controls_mirroring(view, expected_mirror):
    """O gabarito de podoscópio é espelhado de propósito: é usado com a face para cima."""
    result, _, _ = make_pdf(265.0, view=view)
    assert result.placement.mirrored is expected_mirror
    det = float(np.linalg.det(result.placement.matrix))
    assert (det < 0) is expected_mirror
    assert abs(abs(det) - 1.0) < 1e-12


def test_oversized_foot_is_not_scaled_down():
    """Não caber na folha nunca pode virar redução de escala."""
    contour = FS.foot_polygon(320.0, "right", "below")
    lm = FS.expected_landmarks_mm(320.0, "right", "below")
    frame, _ = bootstrap_frame(contour)
    frame = refine_frame(contour, frame, lm["T2"])
    uv = frame.to_local(contour)
    a = frame.to_plane(np.array([[float(np.min(uv[:, 0])), 0.0]]))[0]
    b = frame.to_plane(np.array([[float(np.max(uv[:, 0])), 0.0]]))[0]
    placement = compute_placement(contour, a, b, "below")
    assert not placement.fits_on_page
    assert is_isometry(placement.matrix)

    result = build_foot_pdf(contour_mm=contour, axis_a_mm=a, axis_b_mm=b,
                            laterality="right", measurements={"lengthMm": 320.0})
    assert not result.fits_on_page
    assert result.warnings and "1:1" in result.warnings[0]
    info = inspect_pdf(result.data)
    measured = path_max_caliper_mm(info.paths_with_color(CONTOUR_RGB)[0])
    assert measured == pytest.approx(320.0, abs=0.3)


def test_no_calibration_square_or_ruler_in_output():
    """A folha de fabricação não pode conter régua nem quadrado de calibração."""
    result, _, _ = make_pdf(265.0)
    info = inspect_pdf(result.data)
    contour_ids = {id(p) for p in info.paths_with_color(CONTOUR_RGB)}
    for path in info.paths:
        if id(path) in contour_ids:
            continue
        pts = path.points_mm
        w = float(pts[:, 0].max() - pts[:, 0].min())
        h = float(pts[:, 1].max() - pts[:, 1].min())
        is_50mm_square = abs(w - 50.0) < 1.0 and abs(h - 50.0) < 1.0
        assert not is_50mm_square, "quadrado de 50 mm encontrado no PDF de fabricação"


def test_marker_sheet_is_exactly_50mm():
    """A folha de calibração (outra coisa) deve sair com 50,00 mm exatos."""
    data = build_marker_sheet_pdf(7)
    info = inspect_pdf(data)
    assert info.is_a4()
    assert "markerSizeMm=50.00" in info.metadata.get("/Keywords", "")
    # As marcas de conferência devem distar exatamente 50 mm.
    ticks = [p for p in info.paths if len(p.points_mm) == 2]
    horizontals = [p for p in ticks
                   if abs(p.points_mm[0][1] - p.points_mm[1][1]) < 0.01]
    assert horizontals, "faltam as marcas de conferência"
    span = max(abs(p.points_mm[0][0] - p.points_mm[1][0]) for p in horizontals)
    assert span == pytest.approx(50.0, abs=0.05)
