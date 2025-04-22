import os
import tempfile

from fastapi import Depends, HTTPException, UploadFile
from loguru import logger

from music_predictor_backend.dto.MusicDTO import PredictByModelResponse
from music_predictor_backend.dto.TextDTO import TextFromMP3
from music_predictor_backend.repository.WhisperRepo import WhisperRepo
from music_predictor_backend.settings.settings import config


class TextService:
    def __init__(self, whisper_repo: WhisperRepo = Depends()):
        self._config = config
        self._whisper_repo = whisper_repo

    def upload_dataset(self, csv_file: UploadFile):
        raise NotImplementedError

    async def predict(self, text: str):
        logger.info(f"Text: {text}")
        return PredictByModelResponse(genres=[])

    async def upload_model(self, model: UploadFile) -> dict[str, str]:
        logger.info(f"Text: {await model.read()}")
        return {"model": "uploaded!"}

    async def get_text(self, file: UploadFile):
        if not file.filename.lower().endswith(".mp3"):
            logger.error("Not mp3")
            raise HTTPException(
                status_code=400, detail="Поддерживаются только MP3 файлы"
            )
        self._whisper_repo.load_model()
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                contents = await file.read()
                temp_audio.write(contents)
                temp_audio_path = temp_audio.name
            task = "transcribe"

            result = self._whisper_repo.get_text(temp_audio_path)
            logger.info(f"Text: {result}")

            return TextFromMP3(text=result["text"], language="en", status="success")
        except Exception as e:
            if "temp_audio_path" in locals() and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            logger.error(e)
            raise HTTPException(
                status_code=500, detail=f"Ошибка обработки аудио: {str(e)}"
            )
