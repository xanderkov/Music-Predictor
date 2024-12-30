import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import tqdm


class MultilabelExperiment:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, device="cpu"):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.training_loss_history = []

    def train(self, num_epochs=10):
        """Training loop with loss plotting"""
        self.model.train()

        for _ in tqdm.tqdm(range(num_epochs)):
            total_loss = 0
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            self.training_loss_history.append(avg_loss)

    def validate(self):
        """Validation loop"""
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                all_preds.append((outputs > 0.1).float())
                all_labels.append(batch_y)
        return torch.cat(all_preds, dim=0).cpu(), torch.cat(all_labels, dim=0).cpu()

    @staticmethod
    def report_metrics(y_true, y_pred, labels):
        """Generate classification metrics"""
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        num_classes = y_true.shape[1]

        metrics_table = []
        precision_list, recall_list, f1_list, accuracy_list = [], [], [], []

        for i in range(num_classes):
            y_true_label = y_true[:, i]
            y_pred_label = y_pred[:, i]
            tn, fp, fn, tp = confusion_matrix(y_true_label, y_pred_label, labels=[0, 1]).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp)  if (tn + fp) > 0 else 0
            bal_accuracy = (recall + specificity) / 2
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            accuracy_list.append(bal_accuracy)
            metrics_table.append([labels[i], precision, recall, f1, bal_accuracy])

        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_accuracy = np.mean(accuracy_list)
        metrics_table.append(["Balanced average", avg_precision, avg_recall, avg_f1, avg_accuracy])

        headers = ["Label", "Precision", "Recall", "F1-Score", "Balanced accuracy"]
        logger.info(tabulate(metrics_table, headers=headers, floatfmt=".4f", tablefmt="grid"))


class MultilabelClassifier2D(nn.Module):
    def __init__(self, sequence_length, input_dim, num_classes, num_params=256):
        super(MultilabelClassifier2D, self).__init__()
        self.training_dataset_name = None
        self.fc = nn.Sequential(
            nn.Linear(sequence_length * input_dim, num_params),
            nn.ReLU(),
            nn.Linear(num_params, num_classes),
            nn.Sigmoid()
        )

    def set_dataset_name(self, training_dataset_name:str):
        self.training_dataset_name = training_dataset_name

    def get_dataset_name(self):
        return self.training_dataset_name

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
