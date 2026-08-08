"""Per-browser-session workspace registry.

Each browser tab gets an isolated `WorkspaceManager` so concurrent users never
share spectroscopy trees or crystal structures. A module-level
`global_workspace` singleton remains in `core.workspace` for non-web scripts.
Nothing here touches physics.
"""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Cookie, Response

from tensorspec.core.workspace import WorkspaceManager

SESSION_COOKIE = "ts_session"

# Sessions are dropped after this much inactivity so long-lived servers do not
# accumulate abandoned tensors in memory.
SESSION_TTL_SECONDS = 8 * 60 * 60

SESSION_ROOT = Path.cwd() / "TensorSpec_Workspace" / "sessions"


@dataclass
class Session:
    """One user's isolated workspace."""

    session_id: str
    workspace: WorkspaceManager
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_seen = time.time()


class SessionStore:
    """Thread-safe registry mapping session ids to workspaces."""

    def __init__(self, root: Path = SESSION_ROOT, ttl: int = SESSION_TTL_SECONDS):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._root = root
        self._ttl = ttl

    def _build_workspace(self, session_id: str) -> WorkspaceManager:
        return WorkspaceManager(project_dir=self._root / session_id)

    def get_or_create(self, session_id: str | None) -> Session:
        self._evict_expired()
        with self._lock:
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session

            new_id = secrets.token_urlsafe(24)
            session = Session(session_id=new_id, workspace=self._build_workspace(new_id))
            self._sessions[new_id] = session
            return session

    def _evict_expired(self) -> None:
        cutoff = time.time() - self._ttl
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.last_seen < cutoff]
            for sid in expired:
                del self._sessions[sid]

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)


session_store = SessionStore()


def current_session(
    response: Response,
    ts_session: str | None = Cookie(default=None),
) -> Session:
    """FastAPI dependency returning the caller's session, creating one if needed."""
    session = session_store.get_or_create(ts_session)
    if ts_session != session.session_id:
        response.set_cookie(
            SESSION_COOKIE,
            session.session_id,
            httponly=True,
            samesite="lax",
            max_age=SESSION_TTL_SECONDS,
        )
    return session
