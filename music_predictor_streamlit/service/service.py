import streamlit as st

from music_predictor_streamlit.service.eda import EDA
from music_predictor_streamlit.service.introduction import make_introduction


class Service:
    def __init__(self):
        pass
    
    def start_service(self):
        condition = st.sidebar.selectbox(
            "Выберете этап",
            ("О проекте", "EDA", "Обучить", "Предсказать")
        )

        if condition == "О проекте":
            make_introduction()
        elif condition == "EDA":
            eda = EDA()
            eda.make_eda()
            
