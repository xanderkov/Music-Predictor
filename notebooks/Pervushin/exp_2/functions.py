import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn.functional import cross_entropy



class LyricsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        # длина датасета
        return len(self.data)

    def __getitem__(self, index):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # получение элементов из датасета
        text = self.data.iloc[index]['lyrics']
        label = self.data.iloc[index]['label']

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].squeeze().to(device),
            'attention_mask': encoding['attention_mask'].squeeze().to(device),
            'label': torch.tensor(label, dtype=torch.long).to(device)
        }


def train_model(model, train_loader, test_loader, optimizer, num_epochs=3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        # for batch in train_loader:
        #     input_ids = batch['input_ids'].to(device)
        #     attention_mask = batch['attention_mask'].to(device)
        #     labels = batch['label'].to(device)

        #     #  обновление градиентов в оптимайзере
        #     optimizer.zero_grad()
        #     outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        #     loss = outputs.loss
        #     loss.backward()
        #     optimizer.step()
        #     total_loss += loss.item()

        #     progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            #  обновление градиентов в оптимайзере
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

        test_loss, test_acc, test_f1, _, _ = evaluate(model, test_loader)
        train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")


def evaluate(model, data_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    total_loss = 0
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"eval..."):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            predictions = torch.argmax(outputs.logits, dim=1)

            total_loss += loss.item()
 
            # из torch тензора в numpy-array
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    return total_loss / len(data_loader), accuracy, f1, true_labels, pred_labels


def predict(texts, labels):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    return [list(labels.keys())[p] for p in predictions]