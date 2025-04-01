from fastapi import UploadFile


class TextService:
    def __init__(
        self,
    ):
        raise NotImplementedError

    def upload_dataset(self, csv_file: UploadFile):
        raise NotImplementedError
