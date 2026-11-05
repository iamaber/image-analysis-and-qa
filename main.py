import os

import uvicorn
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.app import app

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
async def serve_application() -> FileResponse:
    return FileResponse("frontend/application.html")


@app.get("/application")
async def redirect_application() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=307)


@app.get("/application.html")
async def redirect_application_html() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=307)


if __name__ == "__main__":
    reload_enabled = os.getenv("RELOAD", "").strip().lower() in {"1", "true", "yes"}
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=reload_enabled)
