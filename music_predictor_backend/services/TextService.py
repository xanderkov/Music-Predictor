from fastapi import File, UploadFile
from loguru import logger

from music_predictor_backend.dto.MusicDTO import PredictByModelResponse


class TextService:
    def __init__(
        self,
    ):
        self._config = None

    def upload_dataset(self, csv_file: UploadFile):
        raise NotImplementedError

    async def predict(self, text: str):
        logger.info(f"Text: {text}")
        return PredictByModelResponse(genres=[])

    async def upload_model(self, model: UploadFile = File(...)) -> dict[str, str]:
        logger.info(f"Text: {await model.read()}")
        return {"model": "uploaded!"}
