import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from loguru import logger
from streamlit.runtime.uploaded_file_manager import UploadedFile

from music_predictor_streamlit.dto.dto import DatasetNameRequest
from music_predictor_streamlit.service.utils import pandas_to_fastapi_json, send_post_request
from music_predictor_streamlit.settings.settings import config


class EDA:
    def __init__(self):
        self._min_genres = config.eda_config.min_num_genres
        self._back_url = f"http://{config.music_model.backend_host}:{config.music_model.backend_port}/"
        self._upload_url = f"{self._back_url}/api/v1/upload_dataset"
        self._eda_url = f"{self._back_url}/api/v1/make_eda"
        self._set_dataset_url = f"{self._back_url}/api/v1/set_dataset_name"
        self._rare_genres = None  # Bad dicision

    @staticmethod
    def transform_json_response_to_dataframe(data: dict) -> pd.DataFrame | None:
        df = None
        try:
            df = pd.DataFrame(data)
            logger.info(f"Successfully converted json to dataframe")
        except ValueError as e:
            error = "Can't parse json data"
            st.error(error)
            logger.error(error)
        return df

    def get_pandas_from_backend(
        self, json_file: UploadedFile, zip_file: UploadedFile
    ) -> pd.DataFrame | None:
        logger.info("Files getted")
        files = {
            "json_file": (json_file.name, json_file.getvalue(), json_file.type),
            "zip_file": (zip_file.name, zip_file.getvalue(), zip_file.type),
        }
        url = self._upload_url
        logger.info(f"Getting bakcend {url}")
        response = requests.post(url, files=files)
        df = None
        if response.status_code == 200:
            logger.info("Success")
            st.success("Файлы загружены на сервер!")
            # st.json(response.json())
            df = self.transform_json_response_to_dataframe(response.json())
        else:
            error = f"Error: {response.json().get('message', 'Unknown error occurred')}"
            st.error(error)
            logger.error(error)
        return df

    @staticmethod
    def plot_genre_distribution(df):
        try:
            genres = (
                df["genres"].str.get_dummies(sep=" ").sum().sort_values(ascending=False)
            )
            genres.plot(kind="bar", figsize=(12, 6), color="skyblue")
            plt.title("Распределение жанров музыки")
            plt.xlabel("Жанр")
            plt.ylabel("Количество")
            plt.xticks(rotation=45)
            st.pyplot(plt)  # type: ignore
        except Exception as e:
            logger.error(e)
            st.error("Can't plot genre distribution")

    def count_rare_genres(self, df):
        genre_counts = df["genres"].str.split(expand=True).stack().value_counts()

        self._rare_genres = genre_counts[genre_counts < self._min_genres].index
        logger.info(f"Rare genres count: {len(self._rare_genres)}")

    def filter_genres(self, text):
        genres = text.split()
        return " ".join([genre for genre in genres if genre not in self._rare_genres])

    def make_smaller_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        self.count_rare_genres(df)
        df["genres"] = df["genres"].apply(self.filter_genres)
        logger.info("Made smaller genres")
        return df

    def _make_smaller_genres_by_backend(self, df: pd.DataFrame) -> pd.DataFrame:
        files = pandas_to_fastapi_json(df)
        url = self._eda_url
        logger.info(f"Getting bakcend {url}")
        response = requests.post(url, files=files)
        new_df = None
        if response.status_code == 200:
            logger.info("Success")
            st.success("Files uploaded successfully!")
            new_df = self.transform_json_response_to_dataframe(response.json())
        else:
            error = f"Error: {response.json().get('message', 'Unknown error occurred')}"
            st.error(error)
            logger.error(error)
        if new_df is None:
            new_df = df
        return new_df
    
    def _set_dataset_name(self, df: pd.DataFrame):
        url = self._set_dataset_url
        logger.info(f"Getting bakcend {url}")
        if st.button("Сохранить датасет"):

            title = st.text_input("Введите название датасета", "Meine_Kleine_Dataseten")
            dataset_name = DatasetNameRequest(name=title)
            res = send_post_request(url, dataset_name.model_dump())
            if res:
                st.success(f"Датасет с именем {title} сохранен")
        
    def create_analytic(self, df: pd.DataFrame):
        st.write("Загруженные данные. HEAD:")
        st.dataframe(df.head())

        st.subheader("Анализ жанров")
        st.subheader("Изначально заданные жанры")

        self.plot_genre_distribution(df)
        df = self.make_smaller_genres(df)
        st.subheader("Уменьшенное количество жанров")

        self.plot_genre_distribution(df)

        self._set_dataset_name(df)

    def make_eda(self) -> pd.DataFrame | None:
        st.title("EDA")

        json_file = st.file_uploader(
            "Загрузите JSON файл вида: "
            "{'0': {'genres': 'soundtrack classical', 'image_path': 'path'}}",
            type="json",
        )
        zip_file = st.file_uploader("Загрузите ZIP файл со спектограммами", type="zip")
        df = None
        if json_file is not None and zip_file is not None:
            df = self.get_pandas_from_backend(json_file, zip_file)

        if df is not None:
            self.create_analytic(df)
        # st.success("Вы удачно прошли этап EDA проходите на этап обучения!")

        return df
