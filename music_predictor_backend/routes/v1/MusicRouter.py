import argparse
import json
from logging import debug
from random import random, randint
import torch
import torch.optim as optim
from music_predictor_backend.services.MusicService import MusicService
from music_predictor_backend.models.multilabel_model import (
    MultilabelClassifier2D,
    MultilabelExperiment,
)
from typing import List
import uuid
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
    ModelsNamesResponse,
    PredictByModelResponse,
    PredictFilenameResponse,
)


MODEL_DIR = f"{DATA_PATH}/models/"
MODEL_METADATA_FILE = os.path.join(DATA_PATH, "model_names.json")

tempRouter = APIRouter(
    tags=["Music"],
    responses={404: {"description": "Not found"}},
    prefix="/api/v1",
)


async def save_model(model, id_number: str) -> str:
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model_path = os.path.join(MODEL_DIR, f"{id_number}.pkl")

    if os.path.exists(model_path):
        raise FileExistsError(f"Model file already exists for id_number: {id_number}")

    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    return model_path


async def generate_id_number():
    return str(uuid.uuid4())


async def save_model_with_retry(model, max_retries: int = 5):
    retries = 0
    while retries < max_retries:
        try:
            id_number = await generate_id_number()
            model_path = await save_model(model, id_number)
            logger.info(f"Model saved successfully with ID: {id_number}")
            return id_number, model_path
        except FileExistsError:
            retries += 1
            logger.info(
                f"Duplicate file detected. Retrying... ({retries}/{max_retries})"
            )

    raise RuntimeError(
        "Failed to save model after multiple attempts due to duplicates."
    )


@tempRouter.post("/upload_dataset")
async def convert_files_to_dataframe(
    json_file: UploadFile = File(...),
    zip_file: UploadFile = File(...),
):
    logger.info("Get files")

    if json_file.content_type != "application/json":
        raise HTTPException(
            status_code=400, detail={"message": "Invalid JSON file type."}
        )
    json_content = json.loads(await json_file.read())
    zip_content = zipfile.ZipFile(io.BytesIO(await zip_file.read()))

    data = []

    for entry in json_content.values():
        try:
            genres = entry["genres"]
            image_path = entry["image_path"]
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail={"message": "Invalid JSON format."}
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"message": str(e)[:1000]})

        try:
            with zip_content.open(image_path) as img_file:
                image = Image.open(img_file)
                image = image.convert("L")
                img_array = np.array(image)
                logger.info(f"Image mode: {image.mode}")
                logger.info(f"Image shape: {img_array.shape}")
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail={"message": "Picture cannot be processed."}
            )
        data.append({"genres": genres, "img": img_array.tolist()})

    df = pd.DataFrame(data)

    if isinstance(df, pd.DataFrame):
        byte_stream = io.BytesIO()
        pickle.dump(df, byte_stream)
        byte_stream.seek(0)
        return StreamingResponse(byte_stream, media_type="application/octet-stream")


@tempRouter.post("/fit_model")
async def fit_model(fit_request: FitRequest) -> FitResponse:
    logger.info(
        f"Starting model training for {fit_request.epochs} epochs with learning rate {fit_request.learning_rate}"
    )

    music_service = MusicService()
    train_loader, val_loader = music_service.load_data(fit_request.dataset_name)

    sequence_length, input_dim, num_classes = music_service.get_datasets_shape()
    model = MultilabelClassifier2D(
        sequence_length=sequence_length, input_dim=input_dim, num_classes=num_classes
    )
    optimizer = optim.Adam(model.parameters(), lr=fit_request.learning_rate)
    criterion = torch.nn.BCELoss()

    experiment = MultilabelExperiment(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cpu",
    )

    experiment.train(num_epochs=fit_request.epochs)
    experiment.model.set_dataset_name(fit_request.dataset_name)

    y_pred, y_true = experiment.validate()
    logger.info("Training ended successfully")
    model_id, _ = await save_model_with_retry(experiment.model)
    return FitResponse(
        y_true=y_true.tolist(),
        y_pred=y_pred.tolist(),
        training_loss_history=experiment.training_loss_history,
        model_number_id=model_id,
    )


@tempRouter.post("/get_labels")
async def get_labels(name: DatasetNameRequest) -> LabelsResponse:
    logger.info("Get files")
    classes_file_path = os.path.join(DATA_PATH, "datasets", name.name, "classes.txt")

    if not os.path.exists(classes_file_path):
        raise FileNotFoundError(f"The classes file does not exist: {classes_file_path}")

    with open(classes_file_path, "r") as file:
        res = LabelsResponse(labels=[line.strip() for line in file.readlines()])

    logger.info(f"{res}")
    return res


@tempRouter.post("/set_dataset_name")
async def set_dataset_name(
    name: str = Form(...), pickled_dataset: UploadFile = File(...)
) -> DatasetNameResponse:
    try:
        dataset_content = await pickled_dataset.read()
        data = pickle.loads(dataset_content)
        logger.info(f"Dataset content received: {data}")
    except (pickle.UnpicklingError, EOFError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid pickled file: {e}")
    logger.info("Dataset received")
    dataset_folder = f"{DATA_PATH}/datasets/{name}"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    file_path = f"{dataset_folder}/{name}.pkl"

    if os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"Dataset '{name}' already exists.")
    else:
        with open(file_path, "wb") as dump_file:
            pickle.dump(data, dump_file)
        return DatasetNameResponse(message=f"Сохранен датасет {name}")


@tempRouter.get("/get_datasets_names")
async def get_datasets_names() -> DatasetNamesResponse:
    datasets_dir = f"{DATA_PATH}/datasets"
    if not os.path.exists(datasets_dir):
        return DatasetNamesResponse(names=[])
    dataset_files = [file for file in os.listdir(datasets_dir)]
    return DatasetNamesResponse(names=dataset_files)


@tempRouter.get("/models_names")
async def get_models_names() -> ModelsNamesResponse:
    all_models = await load_model_metadata()
    return ModelsNamesResponse(names=all_models.keys())


@tempRouter.post("/predict")
async def predict(
    model_name: str = Form(...),
    data: UploadFile = File(...),
) -> PredictByModelResponse:

    try:

        all_models = await load_model_metadata()
        if model_name not in all_models:
            raise HTTPException(status_code=404, detail="Model not found")

        model = await load_model(all_models[model_name])
        image_bytes = await data.read()
        image = Image.open(io.BytesIO(image_bytes))

        image = np.array(image.convert("L"))
        music_service = MusicService()
        image_tensor = (
            torch.tensor(music_service.process_sample(image), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        model.eval()
        with torch.no_grad():
            output = model(image_tensor).numpy()

        threshold = 0.1
        predicted_genres = await get_labels(
            DatasetNameRequest(name=model.get_dataset_name())
        )
        predicted_genres = [
            genre
            for i, genre in enumerate(predicted_genres.labels)
            if output[0][i] > threshold
        ]

        return PredictByModelResponse(genres=predicted_genres)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


async def load_model_metadata():
    if os.path.exists(MODEL_METADATA_FILE):
        with open(MODEL_METADATA_FILE, "r") as f:
            return json.load(f)
    return {}


async def save_model_metadata(all_models):
    with open(MODEL_METADATA_FILE, "w") as f:
        json.dump(all_models, f, indent=4)


@tempRouter.post("/save_model_name")
async def save_model_name(model: ModelNameRequest) -> DatasetNameResponse:
    all_models = await load_model_metadata()

    if model.name in all_models:
        raise HTTPException(
            status_code=400, detail=f"Model name '{model.name}' already exists"
        )
    all_models[model.name] = model.id

    await save_model_metadata(all_models)
    return DatasetNameResponse(
        message=f"Model '{model.name}' with ID {model.id} has been saved."
    )


async def load_model(id_number):
    model_path = os.path.join(MODEL_DIR, f"{id_number}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file with id_number: {id_number} does not exist."
        )

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    return model
