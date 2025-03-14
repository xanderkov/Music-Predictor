import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from IPython.display import display, clear_output


def calculate_class_weights(y_train):
    """
    Calculate class weights for imbalanced data.
    """
    num_samples_per_class = torch.sum(y_train, dim=0)
    total_samples = torch.sum(num_samples_per_class)
    class_weights = total_samples / (len(y_train[0]) * num_samples_per_class)
    return class_weights


class SpectrogramCNN(nn.Module):
    """
    Improved CNN Model with deeper architecture, L2 regularization, and dropout.
    """
    def __init__(self, num_classes):
        super(SpectrogramCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(3, 3), nn.Dropout(0.4),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, 3), nn.Dropout(0.4),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(3, 3), nn.Dropout(0.4),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super(AsymmetricFocalLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, min=self.clip, max=1 - self.clip)
        
        pos_weight = (1 - probs) ** self.gamma_pos * targets
        neg_weight = probs ** self.gamma_neg * (1 - targets)

        loss = - (pos_weight * torch.log(probs) + neg_weight * torch.log(1 - probs))
        return loss.mean()


def train_model(model, train_loader, val_loader, y_train, num_epochs=50, lr=0.001, w_d=1e-4, device='cuda'):
    """
    Trains the model with real-time updating loss & accuracy plots.
    """
    model.to(device)
    class_weights = calculate_class_weights(torch.tensor(y_train).to(device))
#     criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    criterion = AsymmetricFocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_d)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in tqdm.tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for train_images, train_labels in train_loader:
            train_images, train_labels = train_images.to(device), train_labels.to(device)
            optimizer.zero_grad()
            train_outputs = model(train_images)
            loss = criterion(train_outputs, train_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(train_images)
            
            preds = torch.sigmoid(train_outputs) > 0.5
            train_correct += (train_labels * (preds == train_labels)).sum().item()
            train_total += (train_labels == 1.0).sum().item()

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_correct / train_total)
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_labels).item() / len(val_images)
                
                preds = torch.sigmoid(val_outputs) > 0.5
                val_correct += (val_labels * ( preds == val_labels)).sum().item()
                val_total += (val_labels == 1.0).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_correct / val_total)
        
        if epoch and epoch % 5 == 0:
            clear_output(wait=True)
            plt.figure(figsize=(16, 5))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Progress')
            
            plt.subplot(1, 2, 2)
            plt.plot(train_accuracies, label='Train TP ratio')
            plt.plot(val_accuracies, label='Val TP ratio')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Accuracy Progress')
            
            plt.show()
    return model


def evaluate_model(model, test_loader, mlb, device='cuda'):
    """
    Evaluates the model and prints classification metrics.
    """
    model.to(device)
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = torch.sigmoid(model(images))
            predictions = (outputs > 0.5).int()
            y_true.append(labels.cpu().numpy())  
            y_pred.append(predictions.cpu().numpy())

    y_true, y_pred = np.vstack(y_true), np.vstack(y_pred)
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=mlb.classes_))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    return y_true, y_pred
