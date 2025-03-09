import pandas as pd
import streamlit as st
from utils import del_session_elements, find_col_candidates

from music_predictor_backend.settings.settings import config


class TextTrain:
    def __init__(self):
        self._back_url = f"http://{config.music_model.backend_host}:{config.music_model.backend_port}/text"

    def train(self):
        st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
        st.divider()
        st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
        st.markdown("#### 2. Обучение модели")

        agree2 = st.checkbox("Начать обучение моделей", on_change=del_session_elements)

        if agree2:
            self._read_df_class()
        if agree2 and "df_class" in st.session_state:
            self._choose_columns()
        if agree2 and "df_class_2" in st.session_state:
            self._train_dataset()

    def _read_df_class(self):
        df_class = pd.read_csv("df_c.csv")
        st.session_state["df_class"] = df_class

    def _choose_columns(self):
        df_class = st.session_state["df_class"]

        # Выбор нормализованных данных для обучения
        with st.form("form5"):
            st.write("Выбор нормализованных данных для обучения")
            st.dataframe(df_class.head())
            col1, col2 = st.columns(2)
            with col1:
                norm_col = st.selectbox(
                    "Выберите столбец нужного типа нормализации",
                    options=[col for col in df_class if col.endswith("tokens")],
                    index=0,
                )
            with col2:
                cat_col = st.selectbox(
                    "Выберите столбец с уровнем категории",
                    options=list(df_class.columns),
                    index=list(df_class.columns).index(find_col_candidates(df_class)),
                    placeholder=find_col_candidates(df_class),
                )
            with col1:
                VOCAB_SIZE = st.number_input(
                    "Выберите размер словаря",
                    min_value=2000,
                    max_value=30000,
                    value=15000,
                    step=1000,
                    placeholder="Введите число",
                )
            with col2:
                st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
                submitted = st.form_submit_button(
                    "Выбрать", type="primary", use_container_width=True
                )

            if submitted:
                df_class_2 = df_class[~df_class[norm_col].isna()]
                # st.dataframe(df_3, use_container_width = True)
                st.session_state["norm_col"] = norm_col
                st.session_state["cat_col"] = cat_col
                st.session_state["VOCAB_SIZE"] = VOCAB_SIZE
                st.session_state["df_class_2"] = df_class_2

    def _train_dataset(self):
        raise NotImplementedError
