"""Endpoints da API."""

from __future__ import annotations

import logging
import time

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from .. import schemas as S
from ..config import get_settings
from ..image_io import decode_image, encode_png
from ..pdf.builder import CONTOUR_RGB, build_foot_pdf, build_marker_sheet_pdf
from ..pdf.inspect import inspect_pdf, path_axis_length_mm, path_max_caliper_mm
from ..pipeline import PipelineBlocked, analyze
from ..render.annotated import AnnotatedRenderer
from ..security import UploadRejected, new_review_token, validate_upload
from ..storage import store
from .review import ReviewValidationError, landmark_map, recompute, validate_reviewed
from .serialize import quality_to_schema, marker_to_schema, result_to_response, to_np

logger = logging.getLogger("pedigrafia.api")
router = APIRouter(prefix="/api")


# --------------------------------------------------------------------------- infra
@router.get("/health")
def health() -> dict:
    return {"status": "ok", "pipelineVersion": "1.0.0"}


@router.get("/config")
def config() -> dict:
    s = get_settings()
    return {
        "markerSizeMm": s.marker_size_mm,
        "markerDictionary": s.marker_dictionary,
        "rectifiedPxPerMm": s.rectified_px_per_mm,
        "defaultView": s.default_view,
        "maxUploadBytes": s.max_upload_bytes,
        "sessionTtlSeconds": s.session_ttl_seconds,
        "minQualityScore": s.quality.min_total_score,
        "segmenter": s.segmenter,
        "a4": {"widthMm": 210.0, "heightMm": 297.0},
    }


# ------------------------------------------------------------------------- análise
@router.post("/analyze", response_model=S.AnalyzeResponse)
async def analyze_endpoint(
    request: Request,
    image: UploadFile = File(...),
    view: str = Form(default=""),
    shoeSize: str = Form(default=""),
    shoeSizeSystem: str = Form(default="BR"),
    detectCallosities: str = Form(default="true"),
) -> S.AnalyzeResponse:
    settings = get_settings()
    data = await image.read()
    try:
        # Nunca confiamos no filename nem no content-type declarado.
        validate_upload(data, declared_mime=image.content_type)
        decoded = decode_image(data)
    except UploadRejected as exc:
        raise HTTPException(status_code=415 if exc.code == "unsupported_media_type"
                            else 413 if exc.code == "payload_too_large" else 400,
                            detail={"code": exc.code, "message": exc.reason}) from exc
    finally:
        del data   # a imagem original não é mantida além do necessário

    chosen_view = view if view in ("below", "above") else settings.default_view
    try:
        result = analyze(decoded, view=chosen_view,
                         detect_callosity=detectCallosities.lower() != "false")
    except PipelineBlocked as exc:
        # Não é erro do servidor: é o quality gate funcionando.
        return JSONResponse(status_code=422, content={
            "code": "capture_rejected",
            "captureQuality": quality_to_schema(exc.report).model_dump(),
            "marker": marker_to_schema(exc.marker, [], float("inf")).model_dump(),
            "message": exc.report.summary(),
        })
    except Exception as exc:
        logger.exception("falha no pipeline")   # sem conteúdo de imagem no log
        raise HTTPException(status_code=500, detail={
            "code": "pipeline_error",
            "message": "Não foi possível analisar a imagem.",
        }) from exc

    session = store.create()
    store.write_artifact(session.id, "rectified.png",
                         encode_png(result.rectification.image))
    store.set_meta(session.id, "px_per_mm", result.rectification.px_per_mm)
    store.set_meta(session.id, "origin_mm",
                   result.rectification.origin_mm.tolist())
    store.set_meta(session.id, "view", chosen_view)

    if settings.debug_artifacts:
        store.write_artifact(session.id, "mask.png",
                             encode_png(np.dstack([result.segmentation.mask] * 3)))

    shoe = None
    try:
        shoe = float(shoeSize) if shoeSize.strip() else None
    except ValueError:
        shoe = None

    base = str(request.base_url).rstrip("/")
    return result_to_response(
        result, session.id, session.created_at, session.expires_at,
        f"{base}/api/session/{session.id}/rectified.png",
        shoe, shoeSizeSystem or "BR",
    )


@router.get("/session/{session_id}/rectified.png")
def rectified_image(session_id: str) -> Response:
    data = store.read_artifact(session_id, "rectified.png")
    if data is None:
        raise HTTPException(status_code=404, detail={"code": "not_found"})
    store.touch(session_id)
    return Response(content=data, media_type="image/png",
                    headers={"Cache-Control": "private, max-age=600"})


@router.get("/session/{session_id}/debug/{name}")
def debug_artifact(session_id: str, name: str) -> Response:
    if not get_settings().debug_artifacts:
        raise HTTPException(status_code=404, detail={"code": "debug_disabled"})
    data = store.read_artifact(session_id, name)
    if data is None:
        raise HTTPException(status_code=404, detail={"code": "not_found"})
    return Response(content=data, media_type="image/png")


@router.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict:
    return {"deleted": store.delete(session_id)}


# -------------------------------------------------------------------- medição
@router.post("/measure", response_model=S.MeasureResponse)
def measure(body: S.MeasureRequest) -> S.MeasureResponse:
    try:
        return recompute(body.foot, body.view)
    except ReviewValidationError as exc:
        raise HTTPException(status_code=400, detail={
            "code": "invalid_geometry", "message": str(exc)}) from exc


# ----------------------------------------------------- revisão obrigatória
@router.post("/review/approve", response_model=S.ApproveResponse)
def approve(body: S.ApproveRequest) -> S.ApproveResponse:
    session = store.get(body.sessionId)
    if session is None:
        raise HTTPException(status_code=404, detail={
            "code": "session_expired",
            "message": "Sessão expirada. Refaça a captura."})
    if not body.feet:
        raise HTTPException(status_code=400, detail={
            "code": "no_feet", "message": "Nenhum pé enviado para aprovação."})

    measured = []
    try:
        for foot in body.feet:
            validate_reviewed(foot)
            measured.append(recompute(foot, session.meta.get("view", "below")))
    except ReviewValidationError as exc:
        raise HTTPException(status_code=400, detail={
            "code": "invalid_geometry", "message": str(exc)}) from exc

    token = new_review_token()
    session.review_token = token
    session.approved_at = time.time()
    store.set_meta(body.sessionId, "approved_feet",
                   [f.id for f in body.feet])
    return S.ApproveResponse(reviewToken=token, approvedAt=session.approved_at,
                             feet=measured)


def _require_approval(session_id: str, token: str):
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail={
            "code": "session_expired",
            "message": "Sessão expirada. Refaça a captura."})
    if not session.review_token or session.review_token != token:
        # É este bloqueio que torna a revisão manual obrigatória de fato.
        raise HTTPException(status_code=403, detail={
            "code": "review_required",
            "message": "É obrigatório revisar e aprovar a pedigrafia antes de "
                       "exportar.",
        })
    return session


# ----------------------------------------------------------------- exportação
@router.post("/export-pdf")
def export_pdf(body: S.ExportPdfRequest) -> Response:
    session = _require_approval(body.sessionId, body.reviewToken)
    try:
        contour = validate_reviewed(body.foot)
    except ReviewValidationError as exc:
        raise HTTPException(status_code=400, detail={
            "code": "invalid_geometry", "message": str(exc)}) from exc

    recomputed = recompute(body.foot, body.view)
    lms = landmark_map(body.foot)

    result = build_foot_pdf(
        contour_mm=contour,
        axis_a_mm=[body.foot.axis.aMm.x, body.foot.axis.aMm.y],
        axis_b_mm=[body.foot.axis.bMm.x, body.foot.axis.bMm.y],
        laterality=body.foot.laterality,
        measurements=recomputed.measurements.model_dump(),
        landmarks={k: v.tolist() for k, v in lms.items()},
        metatarsal_line_mm=(to_np(recomputed.metatarsalLineMm)
                            if recomputed.metatarsalLineMm else None),
        medial_arch_mm=to_np(body.foot.medialArch.pointsMm)
        if body.foot.medialArch.pointsMm else None,
        lateral_arch_mm=to_np(body.foot.lateralArch.pointsMm)
        if body.foot.lateralArch.pointsMm else None,
        support_zones=[z.model_dump() for z in body.foot.supportZones],
        view=body.view, patient_label=body.patientLabel,
        include_support_zones=body.includeSupportZones,
        include_arches=body.includeArches,
        include_metatarsals=body.includeMetatarsals,
        include_axis=body.includeAxis,
        include_measurements=body.includeMeasurements,
    )
    store.touch(session.id)
    side = "direito" if body.foot.laterality == "right" else "esquerdo"
    length = recomputed.measurements.lengthMm
    headers = {
        "Content-Disposition":
            f'attachment; filename="pedigrafia-{side}-{length:.0f}mm-1x1.pdf"',
        "X-Pedigrafia-Length-Mm": f"{length:.2f}",
        "X-Pedigrafia-Scale": "1:1",
        "X-Pedigrafia-Fits-A4": "true" if result.fits_on_page else "false",
    }
    return Response(content=result.data, media_type="application/pdf", headers=headers)


@router.post("/render-annotated")
def render_annotated(body: S.RenderAnnotatedRequest) -> Response:
    session = store.get(body.sessionId)
    if session is None:
        raise HTTPException(status_code=404, detail={"code": "session_expired"})

    feet = []
    for foot in body.feet:
        try:
            validate_reviewed(foot)
        except ReviewValidationError as exc:
            raise HTTPException(status_code=400, detail={
                "code": "invalid_geometry", "message": str(exc)}) from exc
        measured = recompute(foot, body.view)
        payload = foot.model_dump()
        payload["measurements"] = measured.measurements.model_dump()
        payload["metatarsalLineMm"] = [p.model_dump()
                                       for p in measured.metatarsalLineMm]
        feet.append(payload)

    scale = float(np.clip(body.scalePxPerMm, 3.0, 16.0))
    renderer = AnnotatedRenderer(px_per_mm=scale)

    background = None
    bg_origin = None
    bg_ppm = None
    raw = store.read_artifact(body.sessionId, "rectified.png")
    if raw is not None:
        import cv2

        background = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        bg_origin = np.array(session.meta.get("origin_mm", [0.0, 0.0]))
        bg_ppm = float(session.meta.get("px_per_mm", 6.0))

    canvas = renderer.render(feet, background, bg_origin, bg_ppm,
                             include_callosities=body.includeCallosities)
    store.touch(body.sessionId)
    return Response(content=encode_png(canvas, compression=6),
                    media_type="image/png",
                    headers={"Content-Disposition":
                             'attachment; filename="pedigrafia-anotada.png"'})


# ------------------------------------------------------------------ ferramentas
@router.get("/marker.pdf")
def marker_pdf(markerId: int = 7, dictionary: str = "") -> Response:
    if not (0 <= markerId <= 999):
        raise HTTPException(status_code=400, detail={"code": "bad_marker_id"})
    try:
        data = build_marker_sheet_pdf(markerId, dictionary or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={
            "code": "bad_dictionary", "message": str(exc)}) from exc
    return Response(content=data, media_type="application/pdf", headers={
        "Content-Disposition": 'inline; filename="marcador-50mm.pdf"'})


@router.post("/verify-pdf", response_model=S.PdfVerification)
async def verify_pdf(file: UploadFile = File(...),
                     declaredLengthMm: float = Form(...)) -> S.PdfVerification:
    """QA: relê um PDF gerado e confere a dimensão física do contorno."""
    data = await file.read()
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail={"code": "payload_too_large"})
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=415, detail={"code": "not_a_pdf"})
    try:
        info = inspect_pdf(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail={
            "code": "unreadable_pdf", "message": "PDF ilegível."}) from exc

    paths = info.paths_with_color(CONTOUR_RGB)
    if not paths:
        return S.PdfVerification(
            mediaBoxPt=list(info.media_box_pt), mediaBoxMm=list(info.media_box_mm),
            isA4=info.is_a4(), contourLengthMm=0.0, contourWidthMm=0.0,
            contourBboxMm=[0, 0, 0, 0], declaredLengthMm=declaredLengthMm,
            lengthErrorMm=float("inf"), pathPointCount=0, ok=False,
            message="Contorno plantar não encontrado no PDF.")

    path = max(paths, key=lambda p: len(p.points_mm))
    length = path_max_caliper_mm(path)
    width = path_axis_length_mm(path, np.array([1.0, 0.0]))
    pts = path.points_mm
    error = abs(length - declaredLengthMm)
    ok = info.is_a4() and error <= 1.0
    return S.PdfVerification(
        mediaBoxPt=[round(v, 6) for v in info.media_box_pt],
        mediaBoxMm=[round(v, 6) for v in info.media_box_mm],
        isA4=info.is_a4(), contourLengthMm=round(length, 3),
        contourWidthMm=round(width, 3),
        contourBboxMm=[round(float(pts[:, 0].min()), 3),
                       round(float(pts[:, 1].min()), 3),
                       round(float(pts[:, 0].max()), 3),
                       round(float(pts[:, 1].max()), 3)],
        declaredLengthMm=declaredLengthMm, lengthErrorMm=round(error, 3),
        pathPointCount=len(pts), ok=ok,
        message="Escala 1:1 confirmada." if ok else
                "Divergência dimensional detectada no PDF.",
    )
