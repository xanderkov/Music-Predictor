import streamlit as st
from loguru import logger

from music_predictor_streamlit.service.introduction import make_introduction
from music_predictor_streamlit.service.spectrogram.eda import EDA
from music_predictor_streamlit.service.spectrogram.ModelPredictor import ModelPredictor
from music_predictor_streamlit.service.spectrogram.ModelTrainer import ModelTrainer
from music_predictor_streamlit.service.text.TextEda import TextEDA
from music_predictor_streamlit.service.text.TextTrain import TextTrain


class Service:
    def __init__(self):
        self._about_project = "О проекте"
        self._eda = "Спектрограммы. Датасеты. Сформировать"
        self._train = "Спектрограммы. Обучить модель"
        self._predict_step = "Спектрограммы. Предсказать жанр песни"

        self._eda_text = "Тексты. Датасеты. Сформировать, предобработать"
        self._train_text = "Тексты. Обучить модель"
        self._predict_text = "Тексты. Предсказать жанр"

    def start_service(self):
        condition = st.sidebar.selectbox(
            "Выберете этап",
            (
                self._about_project,
                self._eda,
                self._train,
                self._predict_step,
                self._eda_text,
                self._train_text,
                self._predict_text,
            ),
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

        elif condition == self._eda_text:
            logger.info("Choose eda text")
            text_eda = TextEDA()
            text_eda.make_eda()
        elif condition == self._train_text:
            logger.info("Choose train text")
            text_train = TextTrain()
            text_train.train()
        elif condition == self._predict_text:
            logger.info("Choose predict text")
