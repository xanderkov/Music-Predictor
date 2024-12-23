from random import random

from fastapi import Form, File, UploadFile, FastAPI
from typing import List
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import pandas as pd
import json
from loguru import logger
import argparse
import json

import yaml

from music_predictor_streamlit.dto.dto import FitRequest, FitResponse

app = FastAPI()

async def read_json(json_file: UploadFile = File(...)) -> pd.DataFrame | JSONResponse:
    if json_file.content_type != 'application/json':
        return JSONResponse(status_code=400, content={"message": "Invalid JSON file type."})
    
    json_content = await json_file.read()
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"message": "Invalid JSON format."})
    
    try:
        df = pd.DataFrame(data)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"message": str(e)})
    return df

@app.post("/api/v1/upload_dataset")
async def upload_files(json_file: UploadFile = File(...), zip_file: UploadFile = File(...)):
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
    n = 100
    y_true = [random() for _ in range(n)]
    y_pred = [random() for _ in range(n)]
    training_loss_history = [(100 - i) / 100 for i in range(n)]
    return FitResponse(y_true=y_true, y_pred=y_pred, training_loss_history=training_loss_history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", action="store", default=22448)
    args = parser.parse_args()
    port = int(args.port)

    uvicorn.run(app, host="0.0.0.0", port=port, access_log=False)


if __name__ == "__main__":
    import uvicorn

    main()
