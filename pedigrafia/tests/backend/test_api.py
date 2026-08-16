"""Integração da API, revisão obrigatória e segurança de upload."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pdf.builder import CONTOUR_RGB
from app.pdf.inspect import inspect_pdf, path_max_caliper_mm
from app.security import UploadRejected, is_safe_token, sniff_image, validate_upload
from app.storage import store
from app.synth import generator as G


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
    store.purge_all()


@pytest.fixture(scope="module")
def scene_png():
    return G.render_scene(G.bilateral_scene(258.0, 261.0)).encode_png()


def to_reviewed(foot: dict) -> dict:
    return {
        "id": foot["id"], "laterality": foot["laterality"],
        "contourMm": foot["contourMm"], "axis": foot["axis"],
        "landmarks": foot["landmarks"], "medialArch": foot["medialArch"],
        "lateralArch": foot["lateralArch"], "supportZones": foot["supportZones"],
        "callosityHints": foot["callosityHints"], "toeCutT": foot["toeCutT"],
        "notes": "",
    }


# ------------------------------------------------------------------- segurança


def test_sniffing_ignores_declared_content_type():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    assert sniff_image(png).mime == "image/png"
    with pytest.raises(UploadRejected):
        sniff_image(b"GIF89a" + b"\x00" * 32)


def test_upload_size_limit(monkeypatch):
    from app import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("PEDIGRAFIA_MAX_UPLOAD_BYTES", "1024")
    try:
        big = b"\xff\xd8\xff" + b"\x00" * 4096
        with pytest.raises(UploadRejected) as exc:
            validate_upload(big)
        assert exc.value.code == "payload_too_large"
    finally:
        monkeypatch.delenv("PEDIGRAFIA_MAX_UPLOAD_BYTES", raising=False)
        config.get_settings.cache_clear()


def test_path_traversal_tokens_rejected():
    assert not is_safe_token("../../etc/passwd")
    assert not is_safe_token("a/b")
    assert not is_safe_token("")
    assert is_safe_token("Ab0-_" * 4)


def test_non_image_upload_is_rejected(client):
    res = client.post("/api/analyze",
                      files={"image": ("x.txt", b"nao sou imagem", "image/png")})
    assert res.status_code == 415


def test_session_artifacts_are_isolated(client, scene_png):
    res = client.post("/api/analyze", files={"image": ("f.png", scene_png, "image/png")})
    sid = res.json()["sessionId"]
    assert client.get(f"/api/session/{sid}/rectified.png").status_code == 200
    assert client.get("/api/session/..%2F..%2Fetc/rectified.png").status_code in (404, 400)
    assert client.delete(f"/api/session/{sid}").json() == {"deleted": True}
    assert client.get(f"/api/session/{sid}/rectified.png").status_code == 404


# --------------------------------------------------------------------- análise


def test_analyze_returns_millimetre_geometry(client, scene_png):
    res = client.post("/api/analyze",
                      files={"image": ("f.png", scene_png, "image/png")},
                      data={"shoeSize": "40"})
    assert res.status_code == 200
    body = res.json()
    assert body["footCount"] == 2
    assert body["captureQuality"]["passed"]
    assert body["marker"]["detected"]
    assert body["marker"]["sizeMm"] == 50.0
    assert body["marker"]["roundTripErrorMm"] < 0.3
    for foot in body["feet"]:
        assert len(foot["contourMm"]) >= 40
        assert all(isinstance(p["x"], float) for p in foot["contourMm"])
        assert foot["measurements"]["archIndexBasis"] == "silhouette"
        ids = {lm["id"] for lm in foot["landmarks"]}
        assert {"T1", "T5", "M1", "M5", "H"} <= ids
    assert "compatível" in body["shoeSizeCheck"]["message"]
    client.delete(f"/api/session/{body['sessionId']}")


def test_bad_capture_is_rejected_with_reasons(client):
    scene = G.render_scene(G.bilateral_scene(258.0, 261.0, defocus_px=6))
    res = client.post("/api/analyze",
                      files={"image": ("f.png", scene.encode_png(), "image/png")})
    assert res.status_code == 422
    body = res.json()
    assert body["code"] == "capture_rejected"
    assert body["captureQuality"]["blockers"]
    assert not body["captureQuality"]["passed"]


def test_capture_without_any_scale_reference_is_rejected(client):
    """Sem referência física não há escala — e a mensagem tem de dizer isso.

    O sistema aceita duas origens de escala (alvo impresso ou objeto normalizado),
    então o bloqueio não pode falar só de "marcador": tem de explicar que uma
    fotografia sozinha não contém tamanho e listar os dois caminhos.
    """
    import cv2

    blank = np.full((1200, 900, 3), 235, np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    assert ok
    res = client.post("/api/analyze",
                      files={"image": ("f.png", buf.tobytes(), "image/png")})
    assert res.status_code == 422
    blockers = res.json()["captureQuality"]["blockers"]
    assert any("referência de dimensão conhecida" in b for b in blockers)
    assert any("alvo impresso" in b for b in blockers)


# ------------------------------------------------ revisão obrigatória e export


def test_export_requires_review(client, scene_png):
    """I9: não existe caminho de exportação sem aprovação."""
    body = client.post("/api/analyze",
                       files={"image": ("f.png", scene_png, "image/png")}).json()
    sid = body["sessionId"]
    foot = to_reviewed(body["feet"][0])

    res = client.post("/api/export-pdf",
                      json={"sessionId": sid, "reviewToken": "x" * 24, "foot": foot})
    assert res.status_code == 403
    assert res.json()["detail"]["code"] == "review_required"

    approve = client.post("/api/review/approve",
                          json={"sessionId": sid, "feet": [foot]})
    assert approve.status_code == 200
    token = approve.json()["reviewToken"]

    ok = client.post("/api/export-pdf",
                     json={"sessionId": sid, "reviewToken": token, "foot": foot})
    assert ok.status_code == 200
    assert ok.headers["content-type"] == "application/pdf"
    client.delete(f"/api/session/{sid}")


def test_exported_pdf_matches_measured_length(client, scene_png):
    body = client.post("/api/analyze",
                       files={"image": ("f.png", scene_png, "image/png")}).json()
    sid = body["sessionId"]
    feet = [to_reviewed(f) for f in body["feet"]]
    token = client.post("/api/review/approve",
                        json={"sessionId": sid, "feet": feet}).json()["reviewToken"]

    for foot in feet:
        res = client.post("/api/export-pdf",
                          json={"sessionId": sid, "reviewToken": token, "foot": foot})
        assert res.status_code == 200
        declared = float(res.headers["X-Pedigrafia-Length-Mm"])
        assert res.headers["X-Pedigrafia-Scale"] == "1:1"
        info = inspect_pdf(res.content)
        assert info.is_a4()
        measured = path_max_caliper_mm(info.paths_with_color(CONTOUR_RGB)[0])
        assert measured == pytest.approx(declared, abs=0.2)
    client.delete(f"/api/session/{sid}")


def test_edited_contour_changes_measurement(client, scene_png):
    """Editar o contorno precisa realmente mudar a medida — e chegar ao PDF."""
    body = client.post("/api/analyze",
                       files={"image": ("f.png", scene_png, "image/png")}).json()
    sid = body["sessionId"]
    foot = to_reviewed(body["feet"][0])
    before = client.post("/api/measure", json={"foot": foot, "view": "below"}).json()

    # Estica o contorno 5 mm ao longo de y (o eixo do pé é aproximadamente vertical).
    ys = [p["y"] for p in foot["contourMm"]]
    y_min = min(ys)
    foot["contourMm"] = [
        {"x": p["x"], "y": p["y"] - 5.0 if p["y"] < y_min + 40.0 else p["y"]}
        for p in foot["contourMm"]
    ]
    after = client.post("/api/measure", json={"foot": foot, "view": "below"}).json()
    delta = after["measurements"]["lengthMm"] - before["measurements"]["lengthMm"]
    assert delta == pytest.approx(5.0, abs=0.6), delta

    token = client.post("/api/review/approve",
                        json={"sessionId": sid, "feet": [foot]}).json()["reviewToken"]
    res = client.post("/api/export-pdf",
                      json={"sessionId": sid, "reviewToken": token, "foot": foot})
    info = inspect_pdf(res.content)
    measured = path_max_caliper_mm(info.paths_with_color(CONTOUR_RGB)[0])
    assert measured == pytest.approx(after["measurements"]["lengthMm"], abs=0.3)
    client.delete(f"/api/session/{sid}")


def test_moving_metatarsals_changes_measurement(client, scene_png):
    body = client.post("/api/analyze",
                       files={"image": ("f.png", scene_png, "image/png")}).json()
    foot = to_reviewed(body["feet"][0])
    before = client.post("/api/measure", json={"foot": foot, "view": "below"}).json()
    for lm in foot["landmarks"]:
        if lm["id"] in ("M1", "M5"):
            lm["positionMm"]["y"] -= 8.0
    after = client.post("/api/measure", json={"foot": foot, "view": "below"}).json()
    assert after["measurements"]["heelToMetatarsalLineMm"] != pytest.approx(
        before["measurements"]["heelToMetatarsalLineMm"], abs=1.0)
    client.delete(f"/api/session/{body['sessionId']}")


def test_invalid_geometry_is_refused(client, scene_png):
    body = client.post("/api/analyze",
                       files={"image": ("f.png", scene_png, "image/png")}).json()
    foot = to_reviewed(body["feet"][0])
    foot["contourMm"] = foot["contourMm"][:2]
    res = client.post("/api/measure", json={"foot": foot, "view": "below"})
    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "invalid_geometry"
    client.delete(f"/api/session/{body['sessionId']}")


def test_verify_pdf_endpoint(client, scene_png):
    body = client.post("/api/analyze",
                       files={"image": ("f.png", scene_png, "image/png")}).json()
    sid = body["sessionId"]
    foot = to_reviewed(body["feet"][0])
    token = client.post("/api/review/approve",
                        json={"sessionId": sid, "feet": [foot]}).json()["reviewToken"]
    pdf = client.post("/api/export-pdf",
                      json={"sessionId": sid, "reviewToken": token, "foot": foot})
    declared = pdf.headers["X-Pedigrafia-Length-Mm"]
    res = client.post("/api/verify-pdf",
                      files={"file": ("a.pdf", pdf.content, "application/pdf")},
                      data={"declaredLengthMm": declared})
    assert res.status_code == 200
    payload = res.json()
    assert payload["ok"] and payload["isA4"]
    assert payload["lengthErrorMm"] < 0.3
    client.delete(f"/api/session/{sid}")


def test_marker_sheet_endpoint(client):
    res = client.get("/api/marker.pdf?markerId=7")
    assert res.status_code == 200
    assert res.content.startswith(b"%PDF")


def test_annotated_render(client, scene_png):
    body = client.post("/api/analyze",
                       files={"image": ("f.png", scene_png, "image/png")}).json()
    sid = body["sessionId"]
    feet = [to_reviewed(f) for f in body["feet"]]
    res = client.post("/api/render-annotated",
                      json={"sessionId": sid, "feet": feet, "view": "below"})
    assert res.status_code == 200
    assert res.content.startswith(b"\x89PNG")
    client.delete(f"/api/session/{sid}")


def test_health_and_config(client):
    assert client.get("/api/health").json()["status"] == "ok"
    cfg = client.get("/api/config").json()
    assert cfg["markerSizeMm"] == 50.0
    assert cfg["a4"] == {"widthMm": 210.0, "heightMm": 297.0}
