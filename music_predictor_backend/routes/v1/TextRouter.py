from fastapi import APIRouter, Depends, File, UploadFile

from music_predictor_backend.dto.MusicDTO import PredictByModelResponse
from music_predictor_backend.services.TextService import TextService

textRouter = APIRouter(
    tags=["Text"],
    responses={404: {"description": "Not found"}},
    prefix="/api/v1/text",
)


@textRouter.post("/upload_dataset")
async def convert_files_to_dataframe(
    csv_file: UploadFile = File(...),
    text_service: TextService = Depends(),
):
    return text_service.upload_dataset(csv_file)


@textRouter.post("/predict")
async def predict(
    text: str,
    text_service: TextService = Depends(),
) -> PredictByModelResponse:
    return await text_service.predict(text)


@textRouter.post("/upload_model")
async def upload_model(
    model: UploadFile = File(...),
    text_service: TextService = Depends(),
) -> dict[str, str]:
    return await text_service.upload_model(model)
