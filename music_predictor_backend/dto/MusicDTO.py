from pydantic import BaseModel


class PredictRequest(BaseModel):
    genres: list[str]


class PredictResponse(BaseModel):
    genres: list[str]    


class ListPredictResponses(BaseModel):
    predicted_genres: list[PredictResponse] = []