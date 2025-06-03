from pydantic import BaseModel


class TextFromMP3(BaseModel):
    text: str
    language: str
    status: str
