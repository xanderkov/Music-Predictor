from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.responses import RedirectResponse

from music_predictor_backend.routes.v1.MusicRouter import tempRouter
from music_predictor_backend.src.utils import setup_metrics_utils


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    setup_metrics_utils()
    yield


app = FastAPI(lifespan=app_lifespan)
app.include_router(tempRouter)


@app.get("/")
async def docs_redirect():
    return RedirectResponse(url="/docs")
