from http.client import HTTPException

from fastapi import APIRouter, Depends, File, UploadFile

from music_predictor_backend.dto.MusicDTO import ListPredictResponses, PredictResponse
from music_predictor_backend.services.MusicService import MusicService

tempRouter = APIRouter(
    tags=["Music"],
    responses={404: {"description": "Not found"}},
    prefix="/api/v1",
)


@tempRouter.post("/predict_by_spectorgrams")
async def predict_by_spectorgrams(files: list[UploadFile] = File(...), service: MusicService = Depends()) -> ListPredictResponses:
    return service.predict_by_spectorgrams(files)
