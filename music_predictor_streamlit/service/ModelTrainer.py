from loguru import logger
import pandas as pd
import streamlit as st

from music_predictor_streamlit.dto.dto import FitRequest, FitResponse
from music_predictor_streamlit.service.eda import EDA
from music_predictor_streamlit.service.utils import pandas_to_fastapi_json, read_json_from_backend
from music_predictor_streamlit.settings.settings import config
import requests
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self):
        self._backend_url = f"http://{config.music_model.backend_host}:{config.music_model.backend_port}"
        self._fit_model_url = f"{self._backend_url}/api/v1/fit_model"
    
    @staticmethod 
    def create_report_metrics(y_true: list[float], y_pred: list[float]):
        """Generate classification metrics"""
        # y_true = y_true.numpy()
        # y_pred = y_pred.numpy()
        # num_classes = y_true.shape[1]
        # 
        # metrics_table = []
        # precision_list, recall_list, f1_list, accuracy_list = [], [], [], []
        # 
        # for i in range(num_classes):
        #     y_true_label = y_true[:, i]
        #     y_pred_label = y_pred[:, i]
        #     tn, fp, fn, tp = confusion_matrix(y_true_label, y_pred_label, labels=[0, 1]).ravel()
        #     precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        #     recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        #     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        #     specificity = tn / (tn + fp)  if (tn + fp) > 0 else 0
        #     bal_accuracy = (recall + specificity) / 2
        #     precision_list.append(precision)
        #     recall_list.append(recall)
        #     f1_list.append(f1)
        #     accuracy_list.append(bal_accuracy)
        #     metrics_table.append([label_classes[i], precision, recall, f1, bal_accuracy])
        # 
        # avg_precision = np.mean(precision_list)
        # avg_recall = np.mean(recall_list)
        # avg_f1 = np.mean(f1_list)
        # avg_accuracy = np.mean(accuracy_list)
        # metrics_table.append(["Balanced average", avg_precision, avg_recall, avg_f1, avg_accuracy])
        # 
        # headers = ["Label", "Precision", "Recall", "F1-Score", "Balanced accuracy"]
        # print(tabulate(metrics_table, headers=headers, floatfmt=".4f", tablefmt="grid"))
    
    @staticmethod
    def _draw_loss(training_loss_history: list[float]):
        plt.clf()
        plt.plot(training_loss_history, label="Training Loss", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        st.pyplot(plt)  # type: ignore

    def _fit_on_backend(self, fir_request: FitRequest, df: pd.DataFrame):
        files = pandas_to_fastapi_json(df)
        url = self._fit_model_url
        logger.info(f"Getting bakcend {url}")
        response = requests.post(url, json=fir_request.model_dump())
        res = read_json_from_backend(response, "Model trained")
        try:
            res = FitResponse.model_validate(res)
            self.create_report_metrics(res.y_true, res.y_pred)
            self._draw_loss(training_loss_history=res.training_loss_history)
            st.success("Модель создана и сохранена!")
        except Exception as e:
            logger.error(e)
        

            
    
    def _create_model(self, df: pd.DataFrame):

        st.subheader("Создание модели")
        epochs = st.number_input("Количетсво эпохъ", min_value=10, max_value=1000, value=100)
        learning_rate = st.number_input("Learning rate", min_value=0.0001, max_value=0.99, value=0.01)
        fir_request = FitRequest(epochs=epochs, learning_rate=learning_rate)
        if st.button("Создать модель"):
            self._fit_on_backend(fir_request, df)
     
    def train(self) -> None:
        st.title("Невероятные приключения модели. Обучение")

        json_file = st.file_uploader("Загрузите JSON файл вида: "
                                     "{'0': {'genres': 'soundtrack classical', 'image_path': 'path'}}", type="json")
        zip_file = st.file_uploader("Загрузите ZIP файл со спектограммами", type="zip")
        df = None
        if json_file is not None and zip_file is not None:
            eda = EDA()
            df = eda.get_pandas_from_backend(json_file, zip_file)
        
        if df is not None:
            self._create_model(df)
