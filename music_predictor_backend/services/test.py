import asyncio
from fastapi import UploadFile
import os
from music_predictor_backend.services.MusicService import MusicService


class FakeUploadFile:
    def __init__(self, file_path):
        self.filename = os.path.basename(file_path)
        self.file_path = file_path

    async def read(self):
        with open(self.file_path, "rb") as f:
            return f.read()


async def run_test():
    service = MusicService()
    test_file = FakeUploadFile("sample.mp3")
    result = await service.predict_by_music_file(test_file)
    print("Predicted genres:", result.genres)


if __name__ == "__main__":
    asyncio.run(run_test())
