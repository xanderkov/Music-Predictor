import streamlit as st

from music_predictor_backend.settings.settings import config
from music_predictor_streamlit.service.text.utils import del_session_elements


class TextPredictor:
    def __init__(self):
        self._back_url = f"http://{config.music_model.backend_host}:{config.music_model.backend_port}/text"

    def predict(self):
        st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
        st.divider()
        st.markdown("<p style='font-size:8px'><br></p>", unsafe_allow_html=True)
        st.markdown("#### 3. Предсказание модели")

        # session_keys = ['df_', 'diff_cols', 'mean_cols', 'df_filter', 'df_report']
        agree3 = st.checkbox(
            "Предсказать категорию для текстов", on_change=del_session_elements
        )  # , args=[session_keys]

        if agree3:
            pass
