from loguru import logger
import os

def setup_logger():
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logger.add("logs/app.log", rotation="1 MB")