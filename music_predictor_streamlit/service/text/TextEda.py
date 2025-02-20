import sys

import requests
import streamlit as st
from loguru import logger

from music_predictor_streamlit.settings.settings import config


class TextEDA:
    def __init__(self):
        self._back_url = f"http://{config.music_model.backend_host}:{config.music_model.backend_port}/text"
        self._url_upload_data = self._back_url + "/upload_dataset"

    def _upload_data(self):
        uploaded_file = st.file_uploader("", type="csv", key="file_uploader")
        data_load_state = st.text("Идет загрузка ...")
        response = requests.post(self._url_upload_data, files=uploaded_file)
        if response.status_code == 200:
            df_1 = response.json()  # Pandas чето должен сделать
            data_load_state.text(
                f"1. Данные загружены. Размер: {round(sys.getsizeof(df_1) / (1024 * 1024), 2)}Mb, {df_1.shape[0]} x {df_1.shape[1]}"
            )
            st.session_state["df_1"] = df_1
        else:
            logger.error(f"Status code: {response.status_code}. Text: {response.text}")

    def make_eda(self):
        st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
        st.markdown("#### 1. Предобработка данных")

        self._upload_data()
