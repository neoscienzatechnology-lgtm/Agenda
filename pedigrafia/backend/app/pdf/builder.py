"""Geração do PDF de fabricação — **um pé por folha A4, escala 1:1 real**.

Regras não negociáveis implementadas aqui:

* ``pagesize=A4`` → MediaBox exatamente 595,2755905511812 × 841,8897637795277 pt.
* Cada vértice é convertido individualmente por ``mm × 72 / 25.4``. Não existe
  ``canvas.scale()``, ``fit``, ``bounding box`` nem "ajustar à página" em lugar algum.
* O posicionamento é uma isometria (ver :mod:`app.pdf.geometry_page`).
* Sem régua e sem quadrado de calibração na folha: a escala vem do marcador físico.
* Se o pé não couber, o sistema **não** reduz — marca ``fitsOnPage=false`` e avisa.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from reportlab.lib.colors import Color
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdfcanvas

from ..config import A4_HEIGHT_MM, A4_WIDTH_MM, MM_TO_PT, get_settings
from ..geometry.polygon import as_xy
from .geometry_page import PagePlacement, compute_placement

# Cor reservada EXCLUSIVAMENTE ao contorno plantar. O verificador de PDF localiza o
# path do contorno por esta cor — nenhum outro elemento pode usá-la.
CONTOUR_RGB = (0.04, 0.13, 0.29)
AXIS_RGB = (0.55, 0.6, 0.68)
METATARSAL_RGB = (0.11, 0.45, 0.85)
MEDIAL_ARCH_RGB = (0.09, 0.62, 0.36)
LATERAL_ARCH_RGB = (0.25, 0.55, 0.9)
SUPPORT_RGB = (0.97, 0.51, 0.09)
TEXT_RGB = (0.09, 0.16, 0.29)

PRINT_NOTICE = ("IMPRIMIR EM TAMANHO REAL / 100%. DESATIVAR “AJUSTAR À "
                "PÁGINA”.")


@dataclass
class PdfBuildResult:
    data: bytes
    placement: PagePlacement
    length_mm: float
    fits_on_page: bool
    warnings: list[str]


def _pt(pts_mm, placement: PagePlacement) -> np.ndarray:
    return placement.to_page_pt(as_xy(pts_mm))


def _draw_polygon(c: pdfcanvas.Canvas, pts_pt: np.ndarray, *, close: bool = True,
                  fill: bool = False, stroke: bool = True) -> None:
    if len(pts_pt) < 2:
        return
    path = c.beginPath()
    path.moveTo(float(pts_pt[0, 0]), float(pts_pt[0, 1]))
    for p in pts_pt[1:]:
        path.lineTo(float(p[0]), float(p[1]))
    if close:
        path.close()
    c.drawPath(path, stroke=1 if stroke else 0, fill=1 if fill else 0)


def build_foot_pdf(*, contour_mm, axis_a_mm, axis_b_mm, laterality: str,
                   measurements: dict, landmarks: dict | None = None,
                   metatarsal_line_mm=None, medial_arch_mm=None,
                   lateral_arch_mm=None, support_zones=None,
                   view: str = "below", patient_label: str = "",
                   include_support_zones: bool = True, include_arches: bool = True,
                   include_metatarsals: bool = True, include_axis: bool = True,
                   include_measurements: bool = True) -> PdfBuildResult:
    settings = get_settings()
    contour = as_xy(contour_mm)
    if len(contour) < 3:
        raise ValueError("contorno com menos de 3 pontos")

    placement = compute_placement(contour, axis_a_mm, axis_b_mm, view=view)
    warnings: list[str] = []
    if not placement.fits_on_page:
        warnings.append(
            f"O pé ({placement.content_height_mm:.1f} × "
            f"{placement.content_width_mm:.1f} mm) excede a área útil da folha A4 em "
            f"{placement.overflow_mm:.1f} mm. A escala foi mantida em 1:1 e o contorno "
            f"pode ultrapassar a margem — NUNCA reduza no momento da impressão.")

    buf = io.BytesIO()
    c = pdfcanvas.Canvas(buf, pagesize=A4)
    c.setTitle(f"Pedigrafia 1:1 — pé {'direito' if laterality == 'right' else 'esquerdo'}")
    c.setAuthor(settings.pdf_author)
    c.setSubject("Molde plantar em escala 1:1 — não redimensionar")
    length_mm = float(measurements.get("lengthMm", 0.0))
    c.setKeywords(
        f"scale=1:1; unit=mm; lengthMm={length_mm:.2f}; "
        f"laterality={laterality}; view={view}; "
        f"rotationDeg={placement.rotation_deg:.4f}; mirrored={placement.mirrored}; "
        f"mmToPt={MM_TO_PT:.12f}"
    )

    # ---------------------------------------------------------- zonas de apoio
    if include_support_zones and support_zones:
        for zone in support_zones:
            poly_mm = as_xy(zone.get("polygonMm"))
            if len(poly_mm) < 3:
                continue
            c.saveState()
            c.setFillColor(Color(*SUPPORT_RGB, alpha=0.13))
            c.setStrokeColor(Color(*SUPPORT_RGB, alpha=0.45))
            c.setLineWidth(0.4)
            _draw_polygon(c, _pt(poly_mm, placement), fill=True, stroke=True)
            c.restoreState()

    # ------------------------------------------------------------------ arcos
    if include_arches:
        for pts, rgb, dash in ((medial_arch_mm, MEDIAL_ARCH_RGB, (3, 2)),
                               (lateral_arch_mm, LATERAL_ARCH_RGB, (1.5, 2))):
            pts = as_xy(pts)
            if len(pts) < 2:
                continue
            c.saveState()
            c.setStrokeColorRGB(*rgb)
            c.setLineWidth(0.6)
            c.setDash(list(dash))
            _draw_polygon(c, _pt(pts, placement), close=False)
            c.restoreState()

    # ------------------------------------------------------------------- eixo
    if include_axis:
        c.saveState()
        c.setStrokeColorRGB(*AXIS_RGB)
        c.setLineWidth(0.5)
        c.setDash([4, 3])
        axis_pts = _pt(np.array([axis_a_mm, axis_b_mm], dtype=np.float64), placement)
        _draw_polygon(c, axis_pts, close=False)
        c.restoreState()

    # ---------------------------------------------------- linha e cabeças MT
    if include_metatarsals:
        mt_pts = as_xy(metatarsal_line_mm)
        if len(mt_pts) >= 2:
            c.saveState()
            c.setStrokeColorRGB(*METATARSAL_RGB)
            c.setLineWidth(0.6)
            _draw_polygon(c, _pt(mt_pts, placement), close=False)
            c.restoreState()
        if landmarks:
            c.saveState()
            c.setFillColorRGB(*METATARSAL_RGB)
            c.setStrokeColorRGB(*METATARSAL_RGB)
            c.setFont("Helvetica", 5.5)
            for key in ("M1", "M2", "M3", "M4", "M5"):
                p = landmarks.get(key)
                if p is None:
                    continue
                q = _pt(np.asarray([p], dtype=np.float64), placement)[0]
                c.circle(float(q[0]), float(q[1]), 1.4, stroke=0, fill=1)
                c.drawString(float(q[0]) + 2.4, float(q[1]) - 1.6, key)
            c.restoreState()

    # -------------------------------------------------- CONTORNO (cor reservada)
    c.saveState()
    c.setStrokeColorRGB(*CONTOUR_RGB)
    c.setLineWidth(0.9)
    c.setDash([])
    _draw_polygon(c, _pt(contour, placement), close=True)
    c.restoreState()

    # ------------------------------------------------------------- identificação
    c.saveState()
    c.setFillColorRGB(*TEXT_RGB)
    c.setFont("Helvetica-Bold", 26)
    label = "D" if laterality == "right" else "E"
    c.drawString(12 * MM_TO_PT, (297 - 22) * MM_TO_PT, label)
    c.setFont("Helvetica", 7)
    c.drawString(12 * MM_TO_PT, (297 - 27) * MM_TO_PT,
                 "DIREITO" if laterality == "right" else "ESQUERDO")
    if patient_label:
        c.setFont("Helvetica", 7)
        c.drawString(24 * MM_TO_PT, (297 - 12) * MM_TO_PT, patient_label[:60])
    c.restoreState()

    # ------------------------------------------------------- medidas compactas
    if include_measurements:
        c.saveState()
        c.setFillColorRGB(*TEXT_RGB)
        c.setFont("Helvetica", 6.5)
        rows = [
            ("Comprimento", measurements.get("lengthMm")),
            ("Antepé", measurements.get("forefootWidthMm")),
            ("Mediopé", measurements.get("midfootWidthMm")),
            ("Calcâneo", measurements.get("heelWidthMm")),
            ("Calcâneo→MT", measurements.get("heelToMetatarsalLineMm")),
        ]
        y = 297 - 12.0
        for name, value in rows:
            if value is None:
                continue
            c.drawRightString(198 * MM_TO_PT, y * MM_TO_PT,
                              f"{name}: {float(value):.1f} mm")
            y -= 3.6
        ai = measurements.get("archIndex")
        if ai is not None:
            c.drawRightString(198 * MM_TO_PT, y * MM_TO_PT,
                              f"Índice de arco (projeção): {float(ai):.3f}")
        c.restoreState()

    # ------------------------------------------------------- aviso de impressão
    # Puramente informativo: não desloca nem redimensiona nada do desenho.
    c.saveState()
    c.setFillColorRGB(*TEXT_RGB)
    c.setFont("Helvetica-Bold", 6.5)
    c.drawCentredString(105 * MM_TO_PT, 7.0 * MM_TO_PT, PRINT_NOTICE)
    if not placement.fits_on_page:
        c.setFillColorRGB(0.75, 0.15, 0.1)
        c.setFont("Helvetica-Bold", 6.5)
        c.drawCentredString(105 * MM_TO_PT, 3.6 * MM_TO_PT,
                            "ATENÇÃO: contorno excede a margem — NÃO reduzir.")
    c.restoreState()

    c.showPage()
    c.save()
    return PdfBuildResult(data=buf.getvalue(), placement=placement,
                          length_mm=length_mm,
                          fits_on_page=placement.fits_on_page, warnings=warnings)


def build_target_sheet_pdf(target, page_margin_mm: float = 10.0) -> bytes:
    """Folha(s) imprimíveis do alvo de calibração de vários marcadores.

    Cada marcador sai em **tamanho físico exato** com marcas de conferência e a sua
    posição declarada em milímetros impressa ao lado. Se o alvo for maior que a folha
    A4 — o caso normal, já que a área útil tem ~320 × 480 mm — cada marcador vai em uma
    página, e a folha traz as coordenadas para posicioná-lo na plataforma.

    **A métrica do sistema é a posição FÍSICA dos marcadores na plataforma, não este
    desenho.** Depois de colar, meça as distâncias entre marcadores com trena/paquímetro
    e, se divergirem, corrija o arquivo do alvo — não o contrário.
    """
    from reportlab.lib.utils import ImageReader

    from ..calibration.marker import generate_marker_image
    from ..image_io import encode_png

    settings = get_settings()
    buf = io.BytesIO()
    c = pdfcanvas.Canvas(buf, pagesize=A4)
    c.setTitle(f"Alvo de calibração '{target.name}' — Pedigrafia Digital")
    c.setAuthor(settings.pdf_author)
    span_w, span_h = target.span_mm()
    c.setKeywords(
        f"target={target.name}; markers={len(target.markers)}; "
        f"spanMm={span_w:.2f}x{span_h:.2f}; dictionary={target.dictionary}"
    )

    for index, placement in enumerate(target.markers):
        size_mm = placement.size_mm
        img = generate_marker_image(placement.marker_id, 1200, target.dictionary)
        reader = ImageReader(io.BytesIO(encode_png(np.dstack([img] * 3))))

        x = (A4_WIDTH_MM - size_mm) / 2.0
        y = A4_HEIGHT_MM - 92.0 - size_mm
        c.drawImage(reader, x * MM_TO_PT, y * MM_TO_PT,
                    width=size_mm * MM_TO_PT, height=size_mm * MM_TO_PT,
                    preserveAspectRatio=False, mask=None)

        c.setStrokeColorRGB(0.6, 0.6, 0.6)
        c.setLineWidth(0.3)
        for xx in (x, x + size_mm):
            c.line(xx * MM_TO_PT, (y - 8) * MM_TO_PT, xx * MM_TO_PT, (y - 3) * MM_TO_PT)
        c.line(x * MM_TO_PT, (y - 5.5) * MM_TO_PT,
               (x + size_mm) * MM_TO_PT, (y - 5.5) * MM_TO_PT)

        c.setFillColorRGB(*TEXT_RGB)
        c.setFont("Helvetica-Bold", 13)
        c.drawCentredString(105 * MM_TO_PT, (A4_HEIGHT_MM - 26) * MM_TO_PT,
                            f"Alvo de calibração — marcador {placement.marker_id} "
                            f"({index + 1} de {len(target.markers)})")
        c.setFont("Helvetica", 9)
        lines = [
            PRINT_NOTICE,
            f"Confira com régua: o quadrado deve medir {size_mm:.2f} mm de lado.",
            "",
            f"Posição na plataforma — canto superior esquerdo deste marcador:",
            f"    X = {placement.origin_mm[0]:.1f} mm     "
            f"Y = {placement.origin_mm[1]:.1f} mm",
            f"(origem = canto superior esquerdo da área útil de "
            f"{span_w:.0f} × {span_h:.0f} mm)",
            "",
            "Cole sobre superfície rígida, no MESMO plano da planta do pé.",
            "Mantenha a borda branca (zona de silêncio) ao redor do quadrado.",
            "Depois de colar, meça as distâncias entre marcadores e confira",
            "com as coordenadas acima — é essa medida física que calibra o sistema.",
        ]
        yy = y - 22.0
        for line in lines:
            c.drawCentredString(105 * MM_TO_PT, yy * MM_TO_PT, line)
            yy -= 5.6
        c.showPage()

    c.save()
    del page_margin_mm
    return buf.getvalue()


def build_marker_sheet_pdf(marker_id: int = 7, dictionary: str | None = None) -> bytes:
    """Folha imprimível com o marcador em 50,00 × 50,00 mm exatos.

    Esta folha é a **ferramenta de calibração** — não confundir com o PDF de
    fabricação, onde régua e quadrado de calibração são proibidos.
    """
    from reportlab.lib.utils import ImageReader

    from ..calibration.marker import generate_marker_image
    from ..image_io import encode_png

    settings = get_settings()
    size_mm = settings.marker_size_mm
    img = generate_marker_image(marker_id, 1200, dictionary)
    reader = ImageReader(io.BytesIO(encode_png(np.dstack([img] * 3))))

    buf = io.BytesIO()
    c = pdfcanvas.Canvas(buf, pagesize=A4)
    c.setTitle("Marcador de calibração 50 × 50 mm — Pedigrafia Digital")
    c.setAuthor(settings.pdf_author)
    c.setKeywords(f"markerSizeMm={size_mm:.2f}; dictionary="
                  f"{dictionary or settings.marker_dictionary}; id={marker_id}")

    x = (210.0 - size_mm) / 2.0
    y = 297.0 - 70.0 - size_mm
    c.drawImage(reader, x * MM_TO_PT, y * MM_TO_PT,
                width=size_mm * MM_TO_PT, height=size_mm * MM_TO_PT,
                preserveAspectRatio=False, mask=None)

    # Marcas de conferência: linhas exatamente nas arestas do marcador, para que o
    # usuário possa medir 50,00 mm com régua depois de imprimir.
    c.setStrokeColorRGB(0.6, 0.6, 0.6)
    c.setLineWidth(0.3)
    for xx in (x, x + size_mm):
        c.line(xx * MM_TO_PT, (y - 8) * MM_TO_PT, xx * MM_TO_PT, (y - 3) * MM_TO_PT)
    c.line(x * MM_TO_PT, (y - 5.5) * MM_TO_PT,
           (x + size_mm) * MM_TO_PT, (y - 5.5) * MM_TO_PT)

    c.setFillColorRGB(*TEXT_RGB)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(105 * MM_TO_PT, (297 - 30) * MM_TO_PT,
                        "Marcador de calibração — Pedigrafia Digital")
    c.setFont("Helvetica", 9)
    lines = [
        PRINT_NOTICE,
        f"Após imprimir, confira com régua: o quadrado deve medir "
        f"{size_mm:.2f} mm de lado.",
        "Cole sobre superfície rígida e plana, no mesmo plano da planta do pé.",
        "Mantenha a borda branca ao redor do marcador (zona de silêncio).",
        f"Dicionário: {dictionary or settings.marker_dictionary} · id {marker_id}",
    ]
    yy = y - 20.0
    for line in lines:
        c.drawCentredString(105 * MM_TO_PT, yy * MM_TO_PT, line)
        yy -= 6.0

    c.showPage()
    c.save()
    return buf.getvalue()
