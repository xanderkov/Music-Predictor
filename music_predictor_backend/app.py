from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.responses import RedirectResponse

from music_predictor_backend.routes.v1.MusicRouter import musicRouter
from music_predictor_backend.routes.v1.TextRouter import textRouter
from music_predictor_backend.src.utils import setup_metrics_utils


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    setup_metrics_utils()
    yield


app = FastAPI(lifespan=app_lifespan)
app.include_router(musicRouter)
app.include_router(textRouter)


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")
