import pandas as pd
from requests import Response
from loguru import logger
import streamlit as st



def pandas_to_fastapi_json(df: pd.DataFrame) -> dict:
    json_file = df.to_json()
    files = {
        'data': ("spectograms.json", json_file, 'application/json'),
    }
    return files


def read_json_from_backend(response: Response, success_msg: str) -> dict | None:
    res = None
    if response.status_code == 200:
        logger.info(success_msg)
        st.success(success_msg)
        res = response.json()
    else:
        logger.error(f"{response.text} - {response.status_code}")
        error = f"Error: {response.json().get('message', 'Unknown error occurred')}"
        st.error(error)
        logger.error(error)
    return res
