"""FastAPI application factory for the TensorSpec browser UI.

Serves the static HTML shell and routes API calls into `core/`. No physics and
no rendering logic belongs in this module.

Run locally:
    uvicorn tensorspec.web.server.app:app --reload

Serve to others on the VPN:
    uvicorn tensorspec.web.server.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tensorspec.web.server.routers import arpes as arpes_router
from tensorspec.web.server.routers import crystal as crystal_router
from tensorspec.web.server.routers import dft as dft_router
from tensorspec.web.server.routers import peem as peem_router
from tensorspec.web.server.routers import workspace as workspace_router
from tensorspec.web.server.session import Session, current_session, session_store

WEB_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = WEB_ROOT / "templates"
STATIC_DIR = WEB_ROOT / "static"
SUITES_DIR = TEMPLATES_DIR / "suites"


def create_app() -> FastAPI:
    app = FastAPI(
        title="TensorSpec",
        description="N-dimensional spectroscopic analysis served to the browser.",
        version="0.1.0",
    )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(workspace_router.router)
    app.include_router(crystal_router.router)
    app.include_router(arpes_router.router)
    app.include_router(dft_router.router)
    app.include_router(peem_router.router)

    @app.get("/", include_in_schema=False)
    @app.get("/main_browser.html", include_in_schema=False)
    def main_browser(session: Session = Depends(current_session)) -> FileResponse:
        return FileResponse(TEMPLATES_DIR / "main_browser.html")

    @app.get("/suites/{page}", include_in_schema=False)
    def suite_page(page: str, session: Session = Depends(current_session)) -> FileResponse:
        # Resolve inside the suites directory so a crafted path cannot escape it.
        candidate = (SUITES_DIR / page).resolve()
        if candidate.parent != SUITES_DIR.resolve() or candidate.suffix != ".html":
            raise HTTPException(status_code=404, detail="Unknown suite page.")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail="Unknown suite page.")
        return FileResponse(candidate)

    @app.get("/api/health", tags=["system"])
    def health() -> dict:
        return {"status": "ok", "active_sessions": session_store.active_count}

    return app


app = create_app()
