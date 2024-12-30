import streamlit as st
from loguru import logger

from music_predictor_streamlit.service.ModelPredictor import ModelPredictor
from music_predictor_streamlit.service.eda import EDA
from music_predictor_streamlit.service.introduction import make_introduction
from music_predictor_streamlit.service.ModelTrainer import ModelTrainer


class Service:
    def __init__(self):
        self._about_project = "О проекте"
        self._eda = "Датасеты. Сформировать"
        self._train = "Обучить модель"
        self._predict_step = "Предсказать жанр песни"

    def start_service(self):
        condition = st.sidebar.selectbox(
            "Выберете этап",
            (self._about_project, self._eda, self._train, self._predict_step),
        )
        if condition == self._about_project:
            logger.info("Choose about project")
            make_introduction()
        elif condition == self._eda:
            logger.info("Choose eda step")
            eda = EDA()
            eda.make_eda()
        elif condition == self._train:
            logger.info("Choose train step")

            model_trainer = ModelTrainer()
            model_trainer.train()
        elif condition == self._predict_step:
            logger.info("Choose predict astep")

            model_predictor = ModelPredictor()
            model_predictor.predict()
