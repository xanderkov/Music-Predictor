import argparse
import json

import yaml
from loguru import logger

from music_predictor_backend.app import app


@logger.catch
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", action="store", default=22448)
    args = parser.parse_args()
    port = int(args.port)
    swagger = app.openapi()
    with open("docs/swagger.json", "w") as outfile:
        json.dump(swagger, outfile)
    with open("docs/swagger.yaml", "w") as outfile:
        yaml.dump(swagger, outfile)

    uvicorn.run(app, host="0.0.0.0", port=port, access_log=False)


if __name__ == "__main__":
    import uvicorn

    main()
