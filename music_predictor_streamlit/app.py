import os

from loguru import logger
from loguru_loki_handler import loki_handler
from loguru import logger
from music_predictor_backend.settings.settings import config


def setup_logger():
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logger.add("logs/app.log", rotation="1 MB")


def setup_loki():
    logger.add(
        sink=loki_handler(
            config.logger.loki_url,
            {"application": "MusicStreamlit", "environment": "Develop"},
        ),
        serialize=True,
    )


def setup_metrics_utils():
    setup_loki()
