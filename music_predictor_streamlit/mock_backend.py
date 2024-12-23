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
app = FastAPI()


@app.post("/upload_dataset")
async def upload_files(json_file: UploadFile = File(...), zip_file: UploadFile = File(...)):
    logger.info("Get files")
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
    logger.info(f"Daframe: {df.T.head()}")
    return JSONResponse(content=df.T.to_dict())


@app.post("/make_eda")
async def make_eda(data: UploadFile = File(...)):
    logger.info("Get files")
    if data.content_type != 'application/json':
        return JSONResponse(status_code=400, content={"message": "Invalid JSON file type."})

    json_content = await data.read()
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"message": "Invalid JSON format."})

    try:
        df = pd.DataFrame(data)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"message": str(e)})
    return JSONResponse(content=df.T.to_dict())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", action="store", default=22448)
    args = parser.parse_args()
    port = int(args.port)

    uvicorn.run(app, host="0.0.0.0", port=port, access_log=False)


if __name__ == "__main__":
    import uvicorn

    main()
