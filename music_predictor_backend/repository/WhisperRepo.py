from fastapi import HTTPException
from transformers import pipeline


class WhisperRepo:

    def __init__(self):
        self._whisper_pipeline = None

    def load_model(self):
        """Загрузка модели Whisper"""
        if self._whisper_pipeline is None:
            try:
                self._whisper_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-medium",
                    device="cpu",
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Ошибка загрузки модели Whisper: {str(e)}"
                )

    def get_text(self, temp_audio_path: str) -> dict:
        return self._whisper_pipeline(
            temp_audio_path,
            chunk_length_s=30,
            stride_length_s=5,
            # task="transcribe in en",
        )
