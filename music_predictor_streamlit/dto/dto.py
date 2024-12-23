from pydantic import BaseModel


class FitRequest(BaseModel):
   epochs: int
   learning_rate: float


class FitResponse(BaseModel):
   y_true: list[float]
   y_pred: list[float]
   training_loss_history: list[float]