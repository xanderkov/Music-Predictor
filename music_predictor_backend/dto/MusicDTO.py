from fastapi import Request, UploadFile
from pydantic import BaseModel


class FitRequest(BaseModel):
    epochs: int
    learning_rate: float
    dataset_name: str


class FitResponse(BaseModel):
    y_true: list[list[int]]
    y_pred: list[list[int]]
    training_loss_history: list[float]
    model_number_id: str


class LabelsResponse(BaseModel):
    labels: list[str]


class DatasetNameRequest(BaseModel):
    name: str


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


class PredictByModelResponse(BaseModel):
    genres: list[str]


class ModelNameRequest(BaseModel):
    name: str
    id: str


class MusicEntry(BaseModel):
    genres: str
    image_path: str


class ModelInputs(BaseModel):
    sequence_length: int
    input_dim: int
    num_classes: int
