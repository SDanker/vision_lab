import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import check_secret, get_settings
from .errors import ApiError, api_error_handler
from .deps import SAFE_METHODS, bearer_token
from .routers import admin, auth, huggingface, images, misc, models, night, projects, tabular, tokens, training

CSRF_HEADER = "x-requested-with"
CSRF_VALUE = "vision-lab"


@asynccontextmanager
async def lifespan(_: FastAPI):
    check_secret(get_settings())
    logging.getLogger("uvicorn").info("Vision Lab AI API iniciada")
    yield


_docs = get_settings().api_docs
app = FastAPI(
    title="Vision Lab AI",
    docs_url="/api/docs" if _docs else None,
    redoc_url=None,
    openapi_url="/api/openapi.json" if _docs else None,
    lifespan=lifespan,
)
app.add_exception_handler(ApiError, api_error_handler)


@app.middleware("http")
async def csrf_guard(request: Request, call_next):
    """Las cookies son SameSite=Lax; además toda petición que modifica datos debe llevar
    una cabecera personalizada que un formulario de otro origen no puede enviar.
    Con `Authorization: Bearer` no hay cookie en juego (get_current_user la ignora), así que no hay nada que falsificar."""
    if (
        request.method not in SAFE_METHODS
        and bearer_token(request) is None
        and request.headers.get(CSRF_HEADER) != CSRF_VALUE
    ):
        return JSONResponse(status_code=403, content={"detail": "Falta la cabecera de seguridad.", "code": "CSRF"})
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


for r in (
    misc.router, auth.router, admin.router, projects.router, images.router, training.router, models.router,
    tokens.router, night.router, tabular.router, huggingface.router,
):
    app.include_router(r)
