import os
from fastapi import UploadFile, HTTPException
from loguru import logger
from pathlib import Path
from music_predictor_backend.dto.MusicDTO import PredictResponse, ListPredictResponses
from music_predictor_backend.settings.settings import config
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from fastapi.exceptions import HTTPException


class MusicService:
    def __init__(self):
        self._config = config.music_model
        self._path_music = self._config.path_backend
        # Initialize the model (use a placeholder model for now)
        self.model = self._load_model()

    def _load_model(self):
        # Placeholder for loading a pre-trained model
        model = MultilabelClassifier2D(sequence_length=1024, input_dim=256, num_classes=10)
        model.load_state_dict(torch.load(self._config.model_path))
        model.eval()  # Set to evaluation mode
        return model

    def _predict(self, filename: str) -> PredictResponse:
        # Load the image, preprocess it, extract features, and get predictions
        image = self._load_image(filename)
        descriptors = extract_cv_sift(image)
        features_tensor = torch.tensor(descriptors).float().unsqueeze(0)
        with torch.no_grad():
            predictions = self.model(features_tensor)
        genres = self._decode_predictions(predictions)
        return PredictResponse(genres=genres)

    def _load_image(self, filename: str) -> np.ndarray:
        # Load and preprocess the image
        image_path = os.path.join(self._path_music, filename)
        image = cv2.imread(image_path)
        return preprocess_image(image)

    def _decode_predictions(self, outputs: torch.Tensor) -> List[str]:
        # Decode the model's output to a list of genres (using MultiLabelBinarizer)
        thresholds = 0.1
        predicted_labels = (outputs > thresholds).float()
        mlb = MultiLabelBinarizer()
        genres = mlb.classes_  # assuming labels were fitted before
        predicted_genres = [genres[i] for i in range(len(predicted_labels[0])) if predicted_labels[0][i] == 1]
        return predicted_genres

    def _save_file(self, file: UploadFile) -> PredictResponse | None:
        if file.filename is None:
            return None
        try:
            contents = file.file.read()
            logger.info(f"Saving file: {self._path_music} {file.filename}")
            image_path = os.path.join(self._path_music, file.filename)
            with open(image_path, "wb") as f:
                f.write(contents)
            return self._predict(image_path)
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            file.file.close()

    def predict_by_spectrograms(self, spectrograms: List[UploadFile]) -> ListPredictResponses:
        logger.info(f"{len(spectrograms)} spectrograms loaded")
        predictions = ListPredictResponses()
        for file in spectrograms:
            predict = self._save_file(file)
            if predict is not None:
                predictions.predicted_genres.append(predict)
        return predictions


class MultilabelClassifier2D(nn.Module):
    def __init__(self, sequence_length, input_dim, num_classes, num_params=256):
        super(MultilabelClassifier2D, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(sequence_length * input_dim, num_params),
            nn.ReLU(),
            nn.Linear(num_params, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)