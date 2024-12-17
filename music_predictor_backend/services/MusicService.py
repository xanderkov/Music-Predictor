
from fastapi import UploadFile
from fastapi.exceptions import HTTPException
from loguru import logger
from pathlib import Path

from music_predictor_backend.dto.MusicDTO import PredictResponse, ListPredictResponses
from music_predictor_backend.settings.settings import config


class MusicService:
    def __init__(self):
        self._config = config.music_model
        self._path_music = self._config.path_backend
    
    def _predict(self, filename: str) -> PredictResponse:
        # THIS IS MOCK
        return PredictResponse(genres=["METALL"])
    
    def _save_file(self, file: UploadFile) -> PredictResponse | None:
        if file.filename is None:
            return None
        try:
            contents = file.file.read()
            logger.info(f"Saving file: {self._path_music} {file.filename}")
            image_path = self._path_music + "/" + file.filename
            with open(image_path, 'wb') as f:
                f.write(contents)
            return self._predict(image_path) 
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=e)
        finally:
            file.file.close()
    
    def predict_by_spectorgrams(self, spectograms: list[UploadFile]) -> ListPredictResponses:
        logger.info(f"{len(spectograms)} spectograms loaded")
        predictions = ListPredictResponses()
        for file in spectograms:
            predict = self._save_file(file)
            if predict is not None:
                predictions.predicted_genres.append(predict)
        return predictions


