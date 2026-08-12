"""Decodificação de imagem e normalização de orientação EXIF.

Só a rotação/espelhamento indicado pelo EXIF é aplicado. **Nunca** há reescala,
recorte ou reamostragem nesta etapa: a resolução nativa é a base de toda a metrologia.
Ao reescrever a imagem, todo o EXIF é descartado (privacidade: GPS, dispositivo, data).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np

from .security import UploadRejected

_EXIF_ORIENTATION_TAG = 274


@dataclass(frozen=True)
class DecodedImage:
    bgr: np.ndarray
    exif_orientation: int
    width: int
    height: int

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1_000_000.0


def _read_exif_orientation(data: bytes) -> int:
    try:
        from PIL import Image  # import tardio: só é necessário para o EXIF

        with Image.open(io.BytesIO(data)) as im:
            exif = im.getexif()
            if exif:
                value = exif.get(_EXIF_ORIENTATION_TAG)
                if isinstance(value, int) and 1 <= value <= 8:
                    return value
    except Exception:
        pass
    return 1


def _apply_orientation(bgr: np.ndarray, orientation: int) -> np.ndarray:
    """Aplica a orientação EXIF. Todas as operações são isometrias em grade de pixels."""
    if orientation == 1:
        return bgr
    if orientation == 2:
        return cv2.flip(bgr, 1)
    if orientation == 3:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(bgr, 0)
    if orientation == 5:
        return cv2.flip(cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE), 1)
    if orientation == 6:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    if orientation == 8:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


def _decode_with_pillow(data: bytes) -> np.ndarray | None:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            rgb = np.asarray(im.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def decode_image(data: bytes) -> DecodedImage:
    """Decodifica bytes → BGR já normalizado pela orientação EXIF."""
    orientation = _read_exif_orientation(data)

    buf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        bgr = _decode_with_pillow(data)
    if bgr is None:
        raise UploadRejected(
            "Não foi possível decodificar a imagem. Tente JPEG ou PNG.",
            code="unsupported_media_type",
        )

    bgr = _apply_orientation(bgr, orientation)
    h, w = bgr.shape[:2]
    if w < 320 or h < 320:
        raise UploadRejected("Imagem com resolução insuficiente (mínimo 320 px).")
    return DecodedImage(bgr=bgr, exif_orientation=orientation, width=w, height=h)


def encode_png(bgr: np.ndarray, *, compression: int = 4) -> bytes:
    """Codifica sem qualquer metadado (o encoder do OpenCV não escreve EXIF)."""
    ok, buf = cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), compression])
    if not ok:
        raise RuntimeError("falha ao codificar PNG")
    return buf.tobytes()


def encode_jpeg(bgr: np.ndarray, *, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("falha ao codificar JPEG")
    return buf.tobytes()
