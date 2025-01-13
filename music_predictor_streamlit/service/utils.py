import pandas as pd
import requests
import streamlit as st
from loguru import logger
from requests import Response


def pandas_to_fastapi_json(df: pd.DataFrame) -> dict:
    json_file = df.to_json()
    files = {
        "data": ("spectograms.json", json_file, "application/json"),
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


def send_post_request(url: str, request: dict) -> dict:
    logger.info(f"Sending POST request to {url}")
    response = requests.post(url, json=request)
    if response.status_code == 200:
        logger.info("Success")
        return response.json()
    else:
        error = f"Error: {response.json().get('message', 'Unknown error occurred')}"
        st.error(error)
        logger.error(error)
        return {}
