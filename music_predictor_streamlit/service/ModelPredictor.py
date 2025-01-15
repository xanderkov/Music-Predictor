from logging import log

import pandas as pd
import streamlit as st
from loguru import logger
import requests
from streamlit.runtime.uploaded_file_manager import UploadedFile

from music_predictor_streamlit.settings.settings import config
from music_predictor_streamlit.dto.dto import (
    DatasetNamesResponse,
    ModelsNamesRequest,
    ModelsNamesResponse,
    PredictFilenameResponse,
    PredictByModelResponse,
)


class ModelPredictor:
    def __init__(self):
        self._backend_url = f"http://{config.music_model.backend_host}:{config.music_model.backend_port}"
        self._send_predict_file_url = f"{self._backend_url}/api/v1/save_predict_file"
        self._predict_genre_url = f"{self._backend_url}/api/v1/predict"
        self._get_models_url = f"{self._backend_url}/api/v1/models_names"

    def _send_file(self, file: UploadedFile) -> PredictFilenameResponse | None:
        files = {
            "data": (file.name, file.getvalue(), file.type),
        }
        url = self._send_predict_file_url
        logger.info(f"Getting backend {url}")
        response = requests.post(url, files=files)
        res = None
        if response.status_code != 200:
            st.error("Не получилось отправить файл((")
        else:
            res = PredictFilenameResponse.model_validate(response.json())
        return res

    def _choose_file(self) -> PredictFilenameResponse | None:
        file = st.file_uploader(
            "Загрузите картинку спектограммы ",  # / mp3 / текст песни
            type=[
                "png",
                "jpg",
                "jpeg",
            ],
        )  #  "mp3", "txt"
        return file

    def _choose_model(self) -> str | None:
        logger.info(f"Getting model")
        res = requests.get(self._get_models_url)
        name = None
        if res.status_code == 200:
            names = ModelsNamesResponse.model_validate(res.json())
            name = st.selectbox(
                "Выберите модель",
                names.names,
            )

            st.write("Вы выбрали:", name)
        else:
            st.error("Не получилось получить имя модели")
        return name

    def _predict_model(self, name: str, file: UploadedFile):
        logger.info(f"Predicting {name}")
        model_body = {"model_name": name}
        res = requests.post(
            self._predict_genre_url,
            files={
                "data": (file.name, file.getvalue(), file.type),
            },
            data=model_body,
        )
        if res.status_code != 200:
            logger.info(res.status_code)
            st.error("Не удалось получить предсказание")
        else:
            res = PredictByModelResponse.model_validate(res.json())
            st.subheader("Полученные Жанры")

            df = pd.DataFrame(
                data=res.genres,
                columns=["Жанры"],
                index=[i for i in range(len(res.genres))],
            )
            st.table(df)

    def predict(self):
        st.header("Предсказание жанра")
        name = self._choose_model()
        file = self._choose_file()

        if name is not None and file is not None:
            if st.button("Предсказать жанры"):

                self._predict_model(name, file)
