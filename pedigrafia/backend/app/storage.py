"""Armazenamento temporário de sessão.

Sem banco de dados, sem login, sem histórico. Cada sessão é um diretório com nome
aleatório e TTL curto; um *reaper* em background apaga tudo que expirou. Também há
apagamento explícito via ``DELETE /api/session/{id}``.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import get_settings
from .security import is_safe_token, new_session_id, safe_artifact_name


@dataclass
class Session:
    id: str
    created_at: float
    expires_at: float
    directory: Path
    meta: dict[str, Any] = field(default_factory=dict)
    review_token: Optional[str] = None
    approved_at: Optional[float] = None

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at


class SessionStore:
    """Store em memória + artefatos em disco. Processo único (MVP)."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        self._reaper: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ ciclo de vida
    def start_reaper(self) -> None:
        if self._reaper is not None:
            return
        settings = get_settings()

        def loop() -> None:
            while not self._stop.wait(settings.session_reaper_interval_seconds):
                try:
                    self.purge_expired()
                except Exception:  # pragma: no cover - reaper nunca derruba a app
                    pass

        self._reaper = threading.Thread(target=loop, name="session-reaper", daemon=True)
        self._reaper.start()

    def stop_reaper(self) -> None:
        self._stop.set()
        self._reaper = None

    # ---------------------------------------------------------------------- operações
    def create(self) -> Session:
        settings = get_settings()
        sid = new_session_id()
        now = time.time()
        directory = settings.storage_path / sid
        directory.mkdir(parents=True, exist_ok=True)
        session = Session(
            id=sid,
            created_at=now,
            expires_at=now + settings.session_ttl_seconds,
            directory=directory,
        )
        with self._lock:
            self._sessions[sid] = session
        return session

    def get(self, sid: str) -> Optional[Session]:
        if not is_safe_token(sid):
            return None
        with self._lock:
            session = self._sessions.get(sid)
        if session is None:
            return None
        if session.expired:
            self.delete(sid)
            return None
        return session

    def touch(self, sid: str) -> None:
        settings = get_settings()
        with self._lock:
            session = self._sessions.get(sid)
            if session is not None:
                session.expires_at = time.time() + settings.session_ttl_seconds

    def delete(self, sid: str) -> bool:
        if not is_safe_token(sid):
            return False
        with self._lock:
            session = self._sessions.pop(sid, None)
        if session is None:
            return False
        shutil.rmtree(session.directory, ignore_errors=True)
        return True

    def purge_expired(self) -> int:
        with self._lock:
            expired = [s.id for s in self._sessions.values() if s.expired]
        for sid in expired:
            self.delete(sid)
        return len(expired)

    def purge_all(self) -> None:
        with self._lock:
            ids = list(self._sessions)
        for sid in ids:
            self.delete(sid)

    # ---------------------------------------------------------------------- artefatos
    def write_artifact(self, sid: str, name: str, data: bytes) -> Path:
        session = self.get(sid)
        if session is None:
            raise KeyError(sid)
        path = session.directory / safe_artifact_name(name)
        # Confirma que o caminho resolvido continua dentro do diretório da sessão.
        resolved = path.resolve()
        if not str(resolved).startswith(str(session.directory.resolve())):
            raise ValueError("path traversal detectado")
        resolved.write_bytes(data)
        return resolved

    def read_artifact(self, sid: str, name: str) -> Optional[bytes]:
        session = self.get(sid)
        if session is None:
            return None
        path = (session.directory / safe_artifact_name(name)).resolve()
        if not str(path).startswith(str(session.directory.resolve())):
            return None
        if not path.is_file():
            return None
        return path.read_bytes()

    def set_meta(self, sid: str, key: str, value: Any) -> None:
        session = self.get(sid)
        if session is not None:
            session.meta[key] = value

    def dump_meta(self, sid: str) -> str:
        session = self.get(sid)
        return json.dumps(session.meta if session else {}, ensure_ascii=False)


store = SessionStore()
