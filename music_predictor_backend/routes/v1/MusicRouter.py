import argparse
import json
from logging import debug
from random import random, randint
from typing import List
from PIL import Image
import gzip
import io
import zipfile
import pickle
import numpy as np
import pandas as pd
import yaml
from fastapi import APIRouter, FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from sklearn.preprocessing import MultiLabelBinarizer
import uvicorn
from backend_data import DATA_PATH
import os

from music_predictor_backend.dto.MusicDTO import (
    DatasetNameRequest,
    DatasetNameResponse,
    FitRequest,
    FitResponse,
    LabelsResponse,
    DatasetNamesResponse,
    ModelNameRequest,
    ModelsNamesRequest,
    FileUploadRequest,
    ModelsNamesResponse,
    PredictByModelResponse,
    PredictFilenameResponse,
    PredictByModelRequest,
)

tempRouter = APIRouter(
    tags=["Music"],
    responses={404: {"description": "Not found"}},
    prefix="/api/v1",
)


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


@tempRouter.post("/upload_dataset")
async def convert_files_to_dataframe(
    metadata: FileUploadRequest
):
    logger.info("Get files")

    if metadata.json_file.content_type != "application/json":
        raise HTTPException(
            status_code=400, detail={"message": "Invalid JSON file type."}
        )
    json_content = json.loads(await metadata.json_file.read())
    zip_content = zipfile.ZipFile(io.BytesIO(await metadata.zip_file.read()))

    data = []

    for entry in json_content.values():
        try:
            genres = entry['genres']
            image_path = entry['image_path']
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail={"message": "Invalid JSON format."}
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail={"message": str(e)[:1000]})

        try:
            with zip_content.open(image_path) as img_file:
                image = Image.open(img_file)
                image = image.convert("L")
                img_array = np.array(image)
                print(f"Image mode: {image.mode}")
                print(f"Image shape: {img_array.shape}")
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail={"message": "Picture cannot be processed."}
            )
        data.append({'genres': genres, 'img': img_array.tolist()})

    df = pd.DataFrame(data)

    if isinstance(df, pd.DataFrame):
        byte_stream = io.BytesIO()
        pickle.dump(df, byte_stream)
        byte_stream.seek(0)
        return StreamingResponse(byte_stream, media_type="application/octet-stream")


@tempRouter.post("/make_eda")
async def make_eda(data: UploadFile = File(...)):
    logger.info("Get files")
    df = await read_json(data)
    if isinstance(df, pd.DataFrame):

        return JSONResponse(content=df.to_dict())


@tempRouter.post("/fit_model")
async def fit_model(fit_request: FitRequest) -> FitResponse:
    logger.info("Fit model")
    # df = await read_json(data)
    n = 3
    y_true = [randint(0, 1) for _ in range(n)]
    y_pred = [randint(0, 1) for _ in range(n)]
    training_loss_history = [(100 - i) / 100 for i in range(n)]
    return FitResponse(
        y_true=y_true, y_pred=y_pred, training_loss_history=training_loss_history
    )


@tempRouter.post("/get_labels")
async def get_labels(name: DatasetNameRequest) -> LabelsResponse:
    logger.info("Get files")
    df = {"genres": []}
    df["genres"] = ["rock pop", "merall"]
    all_genres = [genre.split() for genre in df["genres"]]
    logger.info(f"All genres: {len(all_genres)}")
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(all_genres)
    res = LabelsResponse(labels=list(mlb.classes_))
    logger.info(f"{res}")
    return res


@tempRouter.post("/set_dataset_name")
async def set_dataset_name(dataset_request: DatasetNameRequest) -> DatasetNameResponse:
    try:
        data = pickle.loads(dataset_request.pickled_file)
    except pickle.UnpicklingError:
        raise HTTPException(
            status_code=400, detail="Invalid pickled file")

    file_path = f"{DATA_PATH}/datasets/{dataset_request.name}.pkl"

    if os.path.exists(file_path):
        raise HTTPException(
            status_code=400, detail=f"File '{file_path}' already exists.")
    else:
        with open(file_path, 'wb') as dump_file:
            pickle.dump(data, dump_file)
        return DatasetNameResponse(message=f"Сохранен датасет {dataset_request.name}")


@tempRouter.get("/get_datasets_names")
async def get_datasets_names() -> DatasetNamesResponse:
    datasets_dir = f"{DATA_PATH}/datasets"
    if not os.path.exists(datasets_dir):
        return DatasetNamesResponse(names=[])
    dataset_files = [
        os.path.splitext(file)[0]
        for file in os.listdir(datasets_dir)
        if file.endswith(".pkl")
    ]
    return DatasetNamesResponse(names=dataset_files)


@tempRouter.get("/models_names")
async def get_models_names() -> ModelsNamesResponse:
    return ModelsNamesResponse(names=["Meine_Kleine_Modelen", "Дорогая модель"])


@tempRouter.post("/save_predict_file")
async def save_predict_file(data: UploadFile = File(...)) -> PredictFilenameResponse:
    return PredictFilenameResponse(name="Bugaga")


@tempRouter.post("/predict")
async def predict(data: PredictByModelRequest) -> PredictByModelResponse:
    return PredictByModelResponse(genres=["metall", "rock"])


@tempRouter.post("/save_model_name")
async def save_model_name(model: ModelNameRequest) -> DatasetNameResponse:
    return DatasetNameResponse(message=f"Сохранена модель {model.name}")
