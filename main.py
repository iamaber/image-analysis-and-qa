import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.app import app

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# Serve frontend pages
@app.get("/")
async def serve_login():
    return FileResponse("frontend/login_page.html")


@app.get("/application")
async def serve_application():
    return FileResponse("frontend/application.html")


@app.get("/application.html")
async def serve_application_html():
    return FileResponse("frontend/application.html")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
