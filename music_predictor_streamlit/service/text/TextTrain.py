from music_predictor_backend.settings.settings import config


class TextTrain:
    def __init__(self):
        self._back_url = f"http://{config.music_model.backend_host}:{config.music_model.backend_port}/text"

    def train(self):
        raise NotImplementedError
