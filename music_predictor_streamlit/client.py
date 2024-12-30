from loguru import logger

from music_predictor_streamlit.app import setup_logger, setup_metrics_utils
from music_predictor_streamlit.service.introduction import make_introduction
from music_predictor_streamlit.service.Service import Service


def main():
    setup_logger()
    setup_metrics_utils()

    logger.info("Reindex service")
    service = Service()
    service.start_service()


if __name__ == "__main__":
    main()
