import io
import json
import os
import pickle
import uuid
import zipfile

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from loguru import logger
from PIL import Image

from music_predictor_backend.dto.MusicDTO import (
    DatasetNameRequest,
    DatasetNameResponse,
    DatasetNamesResponse,
    FitRequest,
    FitResponse,
    LabelsResponse,
    ModelNameRequest,
    ModelsNamesRequest,
    ModelsNamesResponse,
    PredictByModelResponse,
    PredictFilenameResponse,
)
from music_predictor_backend.models.multilabel_model import (
    MultilabelClassifier2D,
    MultilabelExperiment,
)
from music_predictor_backend.services.MusicService import MusicService

musicRouter = APIRouter(
    tags=["Music"],
    responses={404: {"description": "Not found"}},
    prefix="/api/v1",
)


@musicRouter.post("/upload_dataset")
async def convert_files_to_dataframe(
    json_file: UploadFile = File(...),
    zip_file: UploadFile = File(...),
    music_service: MusicService = Depends(),
) -> JSONResponse:
    if json_file.content_type != "application/json":
        raise HTTPException(
            status_code=400, detail={"message": "Invalid JSON file type."}
        )

    df = await music_service.convert_files_to_dataframe(json_file, zip_file)
    return JSONResponse(content=df.to_dict(orient="records"))


@musicRouter.post("/fit_model")
async def fit_model(
    fit_request: FitRequest, music_service: MusicService = Depends()
) -> FitResponse:
    return await music_service.fit_model(fit_request)


@musicRouter.post("/get_labels")
async def get_labels(name: DatasetNameRequest) -> LabelsResponse:
    logger.info("Get files")
    classes_file_path = os.path.join(DATA_PATH, "datasets", name.name, "classes.txt")

    if not os.path.exists(classes_file_path):
        raise FileNotFoundError(f"The classes file does not exist: {classes_file_path}")

    with open(classes_file_path, "r") as file:
        res = LabelsResponse(labels=[line.strip() for line in file.readlines()])

    logger.info(f"{res}")
    return res


@musicRouter.post("/set_dataset_name")
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


@musicRouter.get("/get_datasets_names")
async def get_datasets_names(
    music_service: MusicService = Depends(),
) -> DatasetNamesResponse:
    return music_service.get_datasets_names()


@musicRouter.get("/models_names")
async def get_models_names() -> ModelsNamesResponse:
    all_models = await load_model_metadata()
    return ModelsNamesResponse(names=all_models.keys())


@musicRouter.post("/predict")
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


@musicRouter.post("/save_model_name")
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
