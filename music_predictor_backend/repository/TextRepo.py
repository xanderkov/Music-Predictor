import torch
from loguru import logger
from transformers import BertForSequenceClassification, BertTokenizer


class TextRepo:

    def __init__(self):
        self._labels = {"rap": 0, "country": 1, "rock": 2, "pop": 3, "rb": 4}
        self._model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased", num_labels=len(self._labels)
        )
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def predict(self, texts: str) -> list[str]:
        logger.info(f"Predicting {len(texts)} texts")

        inputs = self._tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self._device)
        logger.info(f"Make inputs {inputs}")
        with torch.no_grad():
            outputs = self._model(**inputs)
        logger.info(f"{outputs=}")
        logits = outputs.logits
        _ = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predictions = torch.argmax(outputs.logits, dim=1)
        logger.info(f"{predictions=}")
        return [list(self._labels.keys())[p] for p in predictions]
