from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger

from music_predictor_backend.dto.MusicDTO import (
    DatasetNameRequest,
    DatasetNameResponse,
    DatasetNamesResponse,
    FitRequest,
    FitResponse,
    LabelsResponse,
    ModelNameRequest,
    ModelsNamesResponse,
    PredictByModelResponse,
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
async def get_labels(
    name: DatasetNameRequest, music_service: MusicService = Depends()
) -> LabelsResponse:
    logger.info("Get Labels")
    return music_service.get_labels(name)


@musicRouter.get("/get_datasets_names")
async def get_datasets_names(
    music_service: MusicService = Depends(),
) -> DatasetNamesResponse:
    return music_service.get_datasets_names()


@musicRouter.get("/models_names")
async def get_models_names(
    music_service: MusicService = Depends(),
) -> ModelsNamesResponse:
    return await music_service.get_models_names()


@musicRouter.post("/predict")
async def predict(
    model_name: str = Form(...),
    data: UploadFile = File(...),
    music_service: MusicService = Depends(),
) -> PredictByModelResponse:
    return await music_service.predict(model_name, data)


@musicRouter.post("/save_model_name")
async def save_model_name(
    model: ModelNameRequest, music_service: MusicService = Depends()
) -> DatasetNameResponse:
    return await music_service.save_model_name(model)


# @musicRouter.post("/set_dataset_name")
# async def set_dataset_name(
#     name: str = Form(...), pickled_dataset: UploadFile = File(...)
# ) -> DatasetNameResponse:
#     try:
#         dataset_content = await pickled_dataset.read()
#         data = pickle.loads(dataset_content)
#         logger.info(f"Dataset content received: {data}")
#     except (pickle.UnpicklingError, EOFError) as e:
#         raise HTTPException(status_code=400, detail=f"Invalid pickled file: {e}")
#     logger.info("Dataset received")
#     dataset_folder = f"{DATA_PATH}/datasets/{name}"
#     if not os.path.exists(dataset_folder):
#         os.makedirs(dataset_folder)
#
#     file_path = f"{dataset_folder}/{name}.pkl"
#
#     if os.path.exists(file_path):
#         raise HTTPException(status_code=400, detail=f"Dataset '{name}' already exists.")
#     else:
#         with open(file_path, "wb") as dump_file:
#             pickle.dump(data, dump_file)
#         return DatasetNameResponse(message=f"Сохранен датасет {name}")
