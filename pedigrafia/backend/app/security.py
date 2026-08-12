"""Validação de upload, *sniffing* de MIME e utilidades de segurança.

Nunca confiamos no ``filename`` nem no ``content-type`` enviados pelo cliente.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass

from .config import get_settings

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


class UploadRejected(Exception):
    def __init__(self, reason: str, code: str = "upload_rejected"):
        super().__init__(reason)
        self.reason = reason
        self.code = code


@dataclass(frozen=True)
class SniffedImage:
    mime: str
    extension: str


# Assinaturas ("magic bytes") — a única fonte de verdade sobre o tipo do arquivo.
def sniff_image(data: bytes) -> SniffedImage:
    if len(data) < 12:
        raise UploadRejected("Arquivo muito pequeno para ser uma imagem válida.")

    if data[:3] == b"\xff\xd8\xff":
        return SniffedImage("image/jpeg", ".jpg")
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return SniffedImage("image/png", ".png")
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return SniffedImage("image/webp", ".webp")
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"heim", b"heis", b"hevm", b"hevs", b"mif1", b"msf1"):
            return SniffedImage("image/heic", ".heic")
        if brand in (b"avif", b"avis"):
            return SniffedImage("image/avif", ".avif")
    if data[:2] == b"BM":
        return SniffedImage("image/bmp", ".bmp")

    raise UploadRejected(
        "Formato de imagem não reconhecido. Envie JPEG, PNG ou WebP.",
        code="unsupported_media_type",
    )


def validate_upload(data: bytes, *, declared_mime: str | None = None) -> SniffedImage:
    settings = get_settings()
    if not data:
        raise UploadRejected("Arquivo vazio.")
    if len(data) > settings.max_upload_bytes:
        mb = settings.max_upload_bytes / (1024 * 1024)
        raise UploadRejected(
            f"Arquivo excede o limite de {mb:.0f} MB.", code="payload_too_large"
        )
    sniffed = sniff_image(data)
    if sniffed.mime not in settings.allowed_mimes:
        # BMP/AVIF são reconhecidos mas não aceitos por padrão.
        raise UploadRejected(
            f"Tipo {sniffed.mime} não permitido.", code="unsupported_media_type"
        )
    # `declared_mime` é apenas registrado como divergência; nunca é usado para decidir.
    return sniffed


def new_session_id() -> str:
    """Identificador opaco, imprevisível — nunca derivado do nome do arquivo."""
    return secrets.token_urlsafe(18)


def new_review_token() -> str:
    return secrets.token_urlsafe(24)


def is_safe_token(value: str) -> bool:
    """Impede *path traversal*: só aceitamos tokens do nosso próprio alfabeto."""
    return bool(value) and bool(_TOKEN_RE.match(value))


def safe_artifact_name(name: str) -> str:
    """Sanitiza o nome de um artefato interno (nunca vindo do usuário)."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if cleaned.startswith(".") or ".." in cleaned or "/" in cleaned or "\\" in cleaned:
        raise UploadRejected("Nome de artefato inválido.", code="bad_request")
    return cleaned
