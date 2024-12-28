import argparse
import json
from logging import debug
from random import random, randint
from typing import List

import pandas as pd
import yaml
from fastapi import APIRouter, FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer
import uvicorn

from music_predictor_streamlit.dto.dto import FitRequest, FitResponse, LabelsResponse

app = FastAPI()


async def read_json(json_file: UploadFile = File(...)) -> pd.DataFrame | JSONResponse:
    if json_file.content_type != "application/json":
        return JSONResponse(
            status_code=400, content={"message": "Invalid JSON file type."}
        )

    json_content = await json_file.read()
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400, content={"message": "Invalid JSON format."}
        )

    try:
        df = pd.DataFrame(data)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"message": str(e)})
    return df


@app.post("/api/v1/upload_dataset")
async def upload_files(
    json_file: UploadFile = File(...), zip_file: UploadFile = File(...)
):
    logger.info("Get files")
    df = await read_json(json_file)
    if isinstance(df, pd.DataFrame):
        logger.info(f"Daframe: {df.T.head()}")
        return JSONResponse(content=df.T.to_dict())


@app.post("/api/v1/make_eda")
async def make_eda(data: UploadFile = File(...)):
    logger.info("Get files")
    df = await read_json(data)
    if isinstance(df, pd.DataFrame):

        return JSONResponse(content=df.to_dict())


@app.post("/api/v1/fit_model")
async def fit_model(fit_request: FitRequest) -> FitResponse:
    logger.info("Fit model")
    # df = await read_json(data)
    n = 10
    y_true = [randint(0, 1) for _ in range(n)]
    y_pred = [randint(0, 1) for _ in range(n)]
    training_loss_history = [(100 - i) / 100 for i in range(n)]
    return FitResponse(
        y_true=y_true, y_pred=y_pred, training_loss_history=training_loss_history
    )


@app.post("/api/v1/get_labels")
async def get_labels(data: UploadFile = File(...)) -> LabelsResponse:
    logger.info("Get files")
    df = await read_json(data)
    res = df
    if isinstance(df, pd.DataFrame):
        # TODO: сделать проверку на вшивость
        all_genres = [genre.split() for genre in df["genres"]]
        logger.info(f"All genres: {len(all_genres)}")
        mlb = MultiLabelBinarizer()
        y_train = mlb.fit_transform(all_genres)
        res = LabelsResponse(labels=list(mlb.classes_))
        logger.info(f"{res}")
    return res



if __name__ == "__main__":
    uvicorn.run("mock_backend:app", host="0.0.0.0", port=22448, reload=True)
