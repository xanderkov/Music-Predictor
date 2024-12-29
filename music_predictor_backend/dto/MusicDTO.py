from pydantic import BaseModel
from fastapi import UploadFile, Request


class FitRequest(BaseModel):
    epochs: int
    learning_rate: float


class FitResponse(BaseModel):
    y_true: list[int]
    y_pred: list[int]
    training_loss_history: list[float]


class LabelsResponse(BaseModel):
    labels: list[str]


class DatasetNameRequest(BaseModel):
    name: str
    pickled_file: bytes


class FileUploadRequest(BaseModel):
    json_file: UploadFile
    zip_file: UploadFile


class DatasetNameResponse(BaseModel):
    message: str


class DatasetNamesResponse(BaseModel):
    names: list[str]


class ModelsNamesRequest(BaseModel):
    names: list[str]


class ModelsNamesResponse(BaseModel):
    names: list[str]


class PredictFilenameResponse(BaseModel):
    name: str


class PredictByModelRequest(BaseModel):
    filename: str
    model_name: str


class PredictByModelResponse(BaseModel):
    genres: list[str]


class ModelNameRequest(BaseModel):
    name: str
