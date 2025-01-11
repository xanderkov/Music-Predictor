import os
import pickle
import uuid

from loguru import logger

from music_predictor_backend.models.multilabel_model import (
    MultilabelClassifier2D,
    MultilabelExperiment,
)
from music_predictor_backend.settings.settings import config
import torch
from music_predictor_backend.dto.MusicDTO import (
    MusicEntry,
    DatasetNamesResponse,
    FitResponse,
    FitRequest,
    ModelInputs,
)
from torch.utils.data import DataLoader, TensorDataset


class ModelSpecRepo:
    def __init__(self):
        self._config = config.music_model
        self.model_dir = self._config.model_dir

    @staticmethod
    async def _generate_id_number():
        return str(uuid.uuid4())

    async def _save_model(self, model, id_number: str) -> str:
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, f"{id_number}.pkl")

        if os.path.exists(model_path):
            raise FileExistsError(
                f"Model file already exists for id_number: {id_number}"
            )

        with open(model_path, "wb") as model_file:
            pickle.dump(model, model_file)
        return model_path

    async def _save_model_with_retry(self, model, max_retries: int = 5):
        retries = 0
        while retries < max_retries:
            try:
                id_number = await self._generate_id_number()
                model_path = await self._save_model(model, id_number)
                logger.info(f"Model saved successfully with ID: {id_number}")
                return id_number, model_path
            except FileExistsError:
                retries += 1
                logger.info(
                    f"Duplicate file detected. Retrying... ({retries}/{max_retries})"
                )

        raise RuntimeError(
            "Failed to save model after multiple attempts due to duplicates."
        )

    async def fit_model(
        self,
        model_inputs: ModelInputs,
        fit_request: FitRequest,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        model = MultilabelClassifier2D(
            sequence_length=model_inputs.sequence_length,
            input_dim=model_inputs.input_dim,
            num_classes=model_inputs.num_classes,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=fit_request.learning_rate)
        criterion = torch.nn.BCELoss()

        experiment = MultilabelExperiment(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device="cpu",
        )

        experiment.train(num_epochs=fit_request.epochs)
        experiment.model.set_dataset_name(fit_request.dataset_name)

        y_pred, y_true = experiment.validate()
        logger.info("Training ended successfully")
        model_id, _ = await self._save_model_with_retry(experiment.model)
        return FitResponse(
            y_true=y_true.tolist(),
            y_pred=y_pred.tolist(),
            training_loss_history=experiment.training_loss_history,
            model_number_id=model_id,
        )
