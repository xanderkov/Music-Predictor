import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import tqdm
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


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(in_channels // reduction_ratio, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super(ConvBlock, self).__init__()
        self.use_attention = use_attention

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        if use_attention:
            self.ca = ChannelAttention(out_channels)
            self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)

        if self.use_attention:
            x = self.ca(x) * x
            x = self.sa(x) * x

        x = self.pool(x)
        x = self.dropout(x)
        return x

class SpectrogramAttentionCNN(nn.Module):
    def __init__(self, num_classes, use_attention=True):
        super(SpectrogramAttentionCNN, self).__init__()
        self.block1 = ConvBlock(1, 32, use_attention)
        self.block2 = ConvBlock(32, 64, use_attention)
        self.block3 = ConvBlock(64, 128, use_attention)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_model(model, train_loader, val_loader, y_train, num_epochs=50, lr=1e-3, w_d=1e-4, 
                use_scheduler=True, loss_fn='bce', device='cuda'):

    model.to(device)
    y_train_tensor = torch.tensor(y_train).float().to(device)
    class_weights = calculate_class_weights(y_train_tensor)

    if loss_fn == 'focal':
        criterion = AsymmetricFocalLoss()
    elif loss_fn == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        raise ValueError("loss_fn must be 'bce' or 'focal'")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=w_d)
    
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

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
        
        if use_scheduler:
            scheduler.step(val_loss)
        
        if epoch and (epoch % 10 == 0 or epoch < 10):
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
            
        if len(train_losses) > 5 and np.std(train_losses[:5]) / np.mean(train_losses[:5]) <= 1e-2:
            break
    return model


def evaluate_model(model, test_loader, mlb, verbose=1, device='cuda'):
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
    if verbose:
        print("Classification Report:\n", classification_report(y_true, y_pred, target_names=mlb.classes_))
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    return y_true, y_pred


def calculate_weighted_f1(y_true, y_pred):
    """Calculate weighted F1 score for multi-label classification"""
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='weighted')

