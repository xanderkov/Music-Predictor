import io
import os
import zipfile
from functools import partial
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from loguru import logger
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from music_predictor_backend.dto.MusicDTO import (
    DatasetNameRequest,
    DatasetNamesResponse,
    LabelsResponse,
    ModelInputs,
)
from music_predictor_backend.settings.settings import config


class SpecRepo:
    def __init__(self):
        self._config = config.music_model
        self._dataset_dir = self._config.dataset_dir
        self._json_name = "/spectograms.json"
        self._images_path_name = "/images"
        self._classes_name = "classes.txt"
        self._part_size = 16
        self.sequence_length = None
        self.input_dim = None
        self.num_classes = None

    def save_spectrograms(self, dataset_name: str, zip_content: zipfile.ZipFile) -> str:
        logger.info(f"Saving spectrograms for {dataset_name}")
        dataset_path = os.path.join(self._dataset_dir, dataset_name)
        with zip_content as z:
            z.extractall(dataset_path)
        image_path = dataset_path + self._images_path_name
        logger.info(f"Result image path: {image_path}")
        return image_path

    def save_spectorgrams_json(self, dataset_name: str, df: pd.DataFrame) -> str:
        logger.info(f"Saving spectrograms for {dataset_name}")
        dataset_path = os.path.join(self._dataset_dir, dataset_name)
        df.to_json(dataset_path + self._json_name, orient="records")
        logger.info(f"Result Head \n: {df.head()}")
        return dataset_path

    def get_datasets_names(self) -> DatasetNamesResponse:
        if not os.path.exists(self._dataset_dir):
            return DatasetNamesResponse(names=[])
        dataset_files = [file for file in os.listdir(self._dataset_dir)]
        return DatasetNamesResponse(names=dataset_files)

    @staticmethod
    def read_images(df: pd.DataFrame) -> list[np.ndarray]:
        images = []
        for index, row in df.iterrows():
            image = Image.open(row["image"])
            image = image.convert("L")
            img_array = np.array(image)
            images.append(img_array)
        return images

    def _build_features(self, df: pd.DataFrame, nfeatures=1000):
        """
        Build features in parallel using multiprocessing.
        """
        if "image" not in df.columns:
            raise ValueError("The DataFrame must contain an 'image' column.")

        part_df = np.array_split(df, len(df) % self._part_size)
        all_feats = []
        for p in part_df:
            images = self.read_images(p)
            num_workers = min(cpu_count(), len(images))

            with Pool(num_workers) as pool:
                feats = list(
                    tqdm.tqdm(
                        pool.imap(
                            partial(self.process_sample, nfeatures=nfeatures), images
                        ),
                        total=len(images),
                    )
                )
                all_feats.extend(feats)

        return all_feats

    def process_sample(self, sample, nfeatures=1000):
        """
        Process a single sample to extract features using SIFT.
        """
        img = self._preprocess_image(sample)
        descriptors = self._extract_cv_sift(img, nfeatures=nfeatures)
        return descriptors

    def _preprocess_image(self, image):
        """
        Preprocess the image by cropping and converting to grayscale.
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = image.shape
        image_part = 7
        crop_width = width // image_part
        start_x = width // 2 - width // (image_part * 2)
        end_x = start_x + crop_width
        cropped_image = image[:, start_x:end_x]

        return cropped_image

    def _extract_cv_sift(self, image, nfeatures=1000):
        """
        Extract SIFT features from the image.
        """
        if image is None:
            raise ValueError("Error loading empty image")

        sift = cv2.SIFT_create(nfeatures=nfeatures)
        kp, descriptors = sift.detectAndCompute(np.array(image, dtype=np.uint8), None)
        if len(kp) > nfeatures:
            descriptors = descriptors[:nfeatures]
        return descriptors

    def _read_dataset_to_pd(
        self, dataset_folder: str, dataset_name: str
    ) -> pd.DataFrame:
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        data_file_path = f"{dataset_folder}{self._json_name}"
        dataset = pd.read_json(data_file_path).rename({"img": "image"}, axis=1)
        return dataset

    def load_data(
        self, dataset_name: str, batch_size: int = 64
    ) -> [DataLoader, DataLoader]:
        dataset_folder = f"{self._dataset_dir}/{dataset_name}"

        dataset = self._read_dataset_to_pd(dataset_folder, dataset_name)

        X_train, X_test, y_train, y_test = train_test_split(
            dataset[["image"]], dataset["genres"], test_size=0.2, random_state=42
        )
        logger.info(f"Train: {X_train.head()}")
        train_feats = self._build_features(X_train, nfeatures=1000)
        test_feats = self._build_features(X_test, nfeatures=1000)

        mlb = MultiLabelBinarizer()
        y_train = mlb.fit_transform(y_train.apply(lambda row: row.split(" ")))
        y_test_encoder = mlb.transform(y_test.apply(lambda row: row.split(" ")))

        with open(f"{dataset_folder}/{self._classes_name}", "w") as file:
            for genre in mlb.classes_:
                file.write(f"{genre}\n")

        train_feats = list(train_feats)
        test_feats = list(test_feats)
        y_train = list(y_train)
        for i in range(len(train_feats) - 1, -1, -1):
            if train_feats[i].shape[0] != 1000:
                train_feats.pop(i)
                y_train.pop(i)
                train_feats.pop(i)

        y_test_encoder = list(y_test_encoder)
        for i in range(len(test_feats) - 1, -1, -1):
            if test_feats[i].shape[0] != 1000:
                test_feats.pop(i)
                y_test_encoder.pop(i)
                test_feats.pop(i)

        train_feats = np.array(train_feats)
        test_feats = np.array(test_feats)
        y_train = np.array(y_train)
        y_test_encoder = np.array(y_test_encoder)

        num_samples, sequence_length, num_features = train_feats.shape
        (self.sequence_length, self.input_dim, self.num_classes) = (
            sequence_length,
            num_features,
            len(mlb.classes_),
        )
        X_train_flat = train_feats.reshape(num_samples, -1)
        X_val_flat = test_feats.reshape(test_feats.shape[0], -1)

        scaler = StandardScaler()
        X_train_scaled_flat = scaler.fit_transform(X_train_flat)
        X_val_scaled_flat = scaler.transform(X_val_flat)

        X_train_scaled = X_train_scaled_flat.reshape(
            num_samples, sequence_length, num_features
        )
        X_val_scaled = X_val_scaled_flat.reshape(
            test_feats.shape[0], sequence_length, num_features
        )

        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled).float(), torch.tensor(y_train).float()
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled).float(), torch.tensor(y_test_encoder).float()
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader

    def get_datasets_shape(self) -> ModelInputs:
        return ModelInputs(
            sequence_length=self.sequence_length,
            input_dim=self.input_dim,
            num_classes=self.num_classes,
        )

    def get_labels(self, name: DatasetNameRequest) -> LabelsResponse:
        classes_file_path = f"{self._dataset_dir}/{name.name}/{self._classes_name}"
        logger.info(f"Loading labels from {classes_file_path}")
        if not os.path.exists(classes_file_path):
            logger.error("File no Founed")
            raise FileNotFoundError(
                f"The classes file does not exist: {classes_file_path}"
            )

        with open(classes_file_path, "r") as file:
            res = LabelsResponse(labels=[line.strip() for line in file.readlines()])

        logger.info(f"{res}")
        return res

    async def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes))

        image = np.array(image.convert("L"))
        image_tensor = (
            torch.tensor(self.process_sample(image), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return image_tensor
