import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import traceback

from loguru import logger
from sklearn.metrics import confusion_matrix

from music_predictor_backend.dto.MusicDTO import (
    DatasetNameRequest,
    DatasetNameResponse,
    FitRequest,
    FitResponse,
    LabelsResponse,
    DatasetNamesResponse,
    ModelNameRequest,
)
from music_predictor_streamlit.service.eda import EDA
from music_predictor_streamlit.service.utils import (
    pandas_to_fastapi_json,
    read_json_from_backend,
    send_post_request,
)
from music_predictor_streamlit.settings.settings import config


class ModelTrainer:
    def __init__(self):
        self._backend_url = f"http://{config.music_model.backend_host}:{config.music_model.backend_port}"
        self._fit_model_url = f"{self._backend_url}/api/v1/fit_model"
        self._get_labels_url = f"{self._backend_url}/api/v1/get_labels"
        self._get_datasets_names = f"{self._backend_url}/api/v1/get_datasets_names"
        self._save_model_name_url = f"{self._backend_url}/api/v1/save_model_name"
        self._trained_models = {}

    @staticmethod
    def create_report_metrics(
        y_true_list: list[int], y_pred_list: list[int], label_classes: list[str]
    ):
        """Generate classification metrics"""
        y_true = np.asarray(y_true_list)
        y_pred = np.asarray(y_pred_list)

        metrics_table = []
        precision_list, recall_list, f1_list, accuracy_list = [], [], [], []

        for i in range(len(y_true_list)):
            y_true_label = y_true_list
            y_pred_label = y_pred_list
            tn, fp, fn, tp = confusion_matrix(
                y_true_label, y_pred_label, labels=[0, 1]
            ).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            bal_accuracy = (recall + specificity) / 2
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            accuracy_list.append(bal_accuracy)
            metrics_table.append(
                [label_classes[i], precision, recall, f1, bal_accuracy]
            )

        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_accuracy = np.mean(accuracy_list)
        metrics_table.append(
            ["Balanced average", avg_precision, avg_recall, avg_f1, avg_accuracy]
        )

        headers = ["Label", "Precision", "Recall", "F1-Score", "Balanced accuracy"]
        table = pd.DataFrame(
            data=metrics_table,
            index=[i for i in range(len(metrics_table))],
            columns=headers,
        )
        st.table(table)

    @staticmethod
    def _draw_loss(training_loss_history: list[float]):
        plt.clf()
        plt.plot(training_loss_history, label="Training Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        st.pyplot(plt)  # type: ignore

    def _get_labels(self, name: str) -> list[str]:
        url = self._get_labels_url
        logger.info(f"Getting bakcend {url}")
        response = requests.post(url, json=DatasetNameRequest(name=name).model_dump())
        labels = read_json_from_backend(response, "Получены метрики!")
        res = []
        if labels is not None:
            labels = LabelsResponse.model_validate(labels)
            res = labels.labels
        return res

    def _fit_on_backend(self, fit_request: FitRequest, name: str):
        url = self._fit_model_url
        logger.info(f"Getting backend {url}")
        response = requests.post(url, json=fit_request.model_dump())
        res = read_json_from_backend(response, "Модель обучена!")
        try:
            res = FitResponse.model_validate(res)
            self._draw_loss(training_loss_history=res.training_loss_history)
            labels = self._get_labels(name)
            logger.info(f"len labels: {len(labels)}")
            logger.info(f"True, pred: {res}")
            self.create_report_metrics(res.y_true, res.y_pred, labels)
            st.success("Модель создана!")

            return res.model_id
        except Exception as e:
            logger.error(e)
            logger.error(traceback.format_exc())

    def _set_model_name(self, title: str, model_id: str):
        url = self._save_model_name_url
        logger.info(f"Save model {url}")

        model_name = ModelNameRequest(name=title, id=model_id)
        res = send_post_request(url, model_name.model_dump())
        logger.info(f"{res}")
        self._trained_models[title] = model_id
        st.success(f"Модель с именем {title} сохранена")

    def _create_model(self, name: str):

        st.subheader("Создание модели")
        epochs = st.number_input(
            "Количетсво эпох", min_value=10, max_value=1000, value=100
        )
        learning_rate = st.number_input(
            "Learning rate", min_value=0.0001, max_value=0.99, value=0.01
        )
        fit_request = FitRequest(epochs=epochs, learning_rate=learning_rate, dataset_name=name)
        title = st.text_input(
            "Введите название модели. Для сохранения", "Meine_Kleine_Modeleen"
        )

        if st.button("Создать модель"):
            model_id = self._fit_on_backend(fit_request, name)
            self._set_model_name(title, model_id)

    def _get_dataset_name(self) -> str | None:
        logger.info("Get dataset name")
        name = None
        res = requests.get(self._get_datasets_names)
        if res.status_code == 200:
            names = DatasetNamesResponse.model_validate(res.json())
            name = st.selectbox(
                "Выберите датасет",
                names.names,
            )

            st.write("Вы выбрали:", name)
        else:
            st.error("Не получилось получить имя датасета")
        return name

    def train(self) -> None:
        st.title("Невероятные приключения модели. Обучение")
        name = self._get_dataset_name()
        if name is not None:
            self._create_model(name)
