import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

# Ensure repo root is on sys.path even when running `python server/main.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load `.env` early so feature flags / provider keys are available during import time.
load_dotenv(_REPO_ROOT / ".env")

from src.core.settings import settings  # noqa: E402
from src.runtime import aclose_all  # noqa: E402
from src.utils.logger import request_id_var  # noqa: E402
from server.routers import router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Best-effort cleanup for cached runtime singletons.
    await aclose_all()


app = FastAPI(lifespan=lifespan)
# Serve endpoints both at `/` and `/api` so the frontend can work:
# - with Vite dev proxy (rewrites `/api/*` -> `/*`)
# - without any proxy (calls `/api/*` directly)
app.include_router(router)
app.include_router(router, prefix="/api")


# Attach a stable request id to every response (and expose it to logs via contextvar).
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = req_id
    token = request_id_var.set(req_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = req_id
    return response


# CORS settings
raw_origins = (settings.cors.allow_origins or "*").strip()
if raw_origins == "*" or raw_origins == "":
    cors_allow_origins = ["*"]
    # Spec-compliant default: wildcard origin cannot be combined with credentials.
    cors_allow_credentials = False
else:
    cors_allow_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
    cors_allow_credentials = bool(settings.cors.allow_credentials)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
