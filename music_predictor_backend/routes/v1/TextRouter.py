from fastapi import APIRouter, Depends, File, UploadFile

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
