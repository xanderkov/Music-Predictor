from pydantic import BaseModel


class FitRequest(BaseModel):
    epochs: int
    learning_rate: float


class FitResponse(BaseModel):
    y_true: list[int]
    y_pred: list[int]
    training_loss_history: list[float]


class LabelsResponse(BaseModel):
    labels: list[str]