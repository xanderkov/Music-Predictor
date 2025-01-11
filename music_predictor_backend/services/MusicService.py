import io
import json
import zipfile

import pandas as pd
from loguru import logger

from fastapi import Depends, File, HTTPException, UploadFile

from music_predictor_backend.dto.MusicDTO import (
    MusicEntry,
    DatasetNamesResponse,
    FitResponse,
    FitRequest,
)
from music_predictor_backend.repository.ModelSpectogramRepo import ModelSpecRepo
from music_predictor_backend.repository.SpectogramRepo import SpecRepo
from pathlib import Path


class MusicService:
    def __init__(
        self,
        model_spec_repo: ModelSpecRepo = Depends(),
        data_spec_repo: SpecRepo = Depends(),
    ):

        self._model_spec_repo = model_spec_repo
        self._data_spec_repo = data_spec_repo

    @staticmethod
    async def _convert_entry_in_data(entry: dict[str, any]) -> MusicEntry:
        try:
            genres = entry["genres"]
            image_path = entry["image_path"]
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail={"message": "Invalid JSON format."}
            )
        # list_genres = genres.split(" ")

        return MusicEntry(genres=genres, image_path=image_path)

    async def convert_files_to_dataframe(
        self, json_file: UploadFile = File(...), zip_file: UploadFile = File(...)
    ) -> pd.DataFrame:
        """
        Важная ремарка. На данный момент мы не обращаем внимания на путь предоставленный в JSON.
        Мы его заменяем на удобный нам формат, для сохранения.

        Поступает на JSON в виде:
        {"0", {"genres": "sldfjas ads;lfkj", "image_path": "path/0.png"}, ...}
        Ключ должны совпадать с картинками в ZIP
        """
        logger.info("Get files")

        json_content = json.loads(await json_file.read())
        zip_content = zipfile.ZipFile(io.BytesIO(await zip_file.read()))

        dataset_name = Path(json_file.filename).stem

        specs_path = self._data_spec_repo.save_spectrograms(dataset_name, zip_content)
        data = []

        for key, value in json_content.items():
            m_en = await self._convert_entry_in_data(value)
            data.append({"genres": m_en.genres, "img": f"{specs_path}/{key}.png"})
        df = pd.DataFrame(data)
        self._data_spec_repo.save_spectorgrams_json(dataset_name, df)
        return df

    def get_datasets_names(self) -> DatasetNamesResponse:
        return self._data_spec_repo.get_datasets_names()

    async def fit_model(self, fit_request: FitRequest) -> FitResponse:
        logger.info(
            f"Starting model training for {fit_request.epochs} epochs with learning rate {fit_request.learning_rate}"
        )

        train_loader, val_loader = self._data_spec_repo.load_data(
            fit_request.dataset_name
        )
        logger.info("Getting dataset shape")
        model_inputs = self._data_spec_repo.get_datasets_shape()
        logger.info("Fit model")
        return await self._model_spec_repo.fit_model(
            model_inputs, fit_request, train_loader, val_loader
        )
