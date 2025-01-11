from loguru import logger
from loguru_loki_handler import loki_handler

from music_predictor_backend.settings.settings import config


def setup_loki():
    logger.add(
        sink=loki_handler(
            config.logger.loki_url,
            {"application": "MusicBackend", "environment": "Develop"},
        ),
        serialize=True,
    )


def setup_metrics_utils():
    setup_loki()
