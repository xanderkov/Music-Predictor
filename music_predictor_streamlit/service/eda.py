import streamlit as st
from music_predictor_streamlit.settings.settings import config
import requests


def make_eda():
    st.title("EDA")

    json_file = st.file_uploader("Загрузите JSON файл с данными", type="json")
    zip_file = st.file_uploader("Загрузите ZIP файл с изображениями", type="zip")
    
    if json_file is not None and zip_file is not None:
        files = {
            'json_file': (json_file.name, json_file.getvalue(), json_file.type),
            'zip_file': (zip_file.name, zip_file.getvalue(), zip_file.type),
        }
        url = f"http://{config.music_model.backend_host}:{config.music_model.backend_port}/upload_dataset"
        
        response = requests.post(url, files=files)
        if response.status_code == 200:
            st.success("Files uploaded successfully!")
            st.json(response.json()) 
        else:
            st.error(f"Error: {response.json().get('message', 'Unknown error occurred')}")
    