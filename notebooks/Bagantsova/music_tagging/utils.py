import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import itertools
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import torch
from model.att_cnn import SpectrogramAttentionCNN, train_model, evaluate_model


GENRE_MAP = {
    # Pop
    '60s': 'pop', '70s': 'pop', '80s': 'pop', '90s': 'pop',
    'advertising': 'pop', 'commercial': 'pop', 'corporate': 'pop',
    'dance': 'pop', 'disco': 'pop', 'electropop': 'pop',
    'instrumentalpop': 'pop', 'party': 'pop', 'pop': 'pop',
    'popfolk': 'pop', 'poprock': 'pop', 'positive': 'pop',
    'retro': 'pop', 'synthpop': 'pop',
    
    # Rock
    'alternativerock': 'rock', 'grunge': 'rock', 'hardrock': 'rock',
    'instrumentalrock': 'rock', 'metal': 'rock', 'postrock': 'rock',
    'progressive': 'rock', 'psychedelic': 'rock', 'punkrock': 'rock',
    'rock': 'rock', 'rocknroll': 'rock',
    
    # Hip-Hop / R&B
    'beat': 'rb', 'breakbeat': 'rb', 'drumnbass': 'rb', 'dubstep': 'rb',
    'electronic': 'rb', 'funk': 'rb', 'hiphop': 'rap', 'house': 'rb',
    'rnb': 'rb', 'rap': 'rap', 'soul': 'rb', 'triphop': 'rb',
    
    # Jazz / Blues
    'acidjazz': 'jazz', 'blues': 'jazz', 'bossanova': 'jazz',
    'fusion': 'jazz', 'jazz': 'jazz', 'jazzfunk': 'jazz',
    'lounge': 'jazz', 'swing': 'jazz',
    
    # Classical
    'accordion': 'classical', 'ambient': 'classical', 'ambiental': 'classical',
    'cello': 'classical', 'choir': 'classical', 'classical': 'classical',
    'classicalguitar': 'classical', 'flute': 'classical', 'harp': 'classical',
    'medieval': 'classical', 'orchestra': 'classical', 'orchestral': 'classical',
    'organ': 'classical', 'piano': 'classical', 'pipeorgan': 'classical',
    'strings': 'classical', 'symphonic': 'classical', 'viola': 'classical',
    'violin': 'classical',
    
    # Country / Folk
    'acousticbassguitar': 'country', 'acousticguitar': 'country',
    'banjo': 'country', 'ballad': 'country', 'country': 'country',
    'folk': 'country', 'harmonica': 'country', 'singersongwriter': 'country',
    'ukulele': 'country',
    
    # Electronic / EDM
    'deephouse': 'electronic', 'edm': 'electronic', 'eurodance': 'electronic',
    'minimal': 'electronic', 'newwave': 'electronic', 'synthesizer': 'electronic',
    'techno': 'electronic', 'trance': 'electronic',
    
    # World Music
    'african': 'world', 'bongo': 'world', 'celtic': 'world', 'ethno': 'world',
    'latin': 'world', 'oriental': 'world', 'reggae': 'world', 'ska': 'world',
    'tribal': 'world', 'world': 'world', 'worldfusion': 'world',
    
    # Soundtrack / Atmospheric
    'action': 'soundtrack', 'adventure': 'soundtrack', 'background': 'soundtrack',
    'calm': 'soundtrack', 'cinematic': 'soundtrack', 'documentary': 'soundtrack',
    'drama': 'soundtrack', 'dramatic': 'soundtrack', 'epic': 'soundtrack',
    'film': 'soundtrack', 'hopeful': 'soundtrack', 'melancholic': 'soundtrack',
    'meditative': 'soundtrack', 'motivational': 'soundtrack', 'movie': 'soundtrack',
    'nature': 'soundtrack', 'newage': 'soundtrack', 'romantic': 'soundtrack',
    'sad': 'soundtrack', 'soundscape': 'soundtrack', 'soundtrack': 'soundtrack',
    'space': 'soundtrack', 'sport': 'soundtrack', 'trailer': 'soundtrack',
    'travel': 'soundtrack',
}


def visualize_attention(model, spectrogram, device='cpu'):
    model.eval()
    model.to(device)
    spectrogram = spectrogram.to(device)

    with torch.no_grad():
        x = spectrogram

        x = model.block1.conv(x)
        ca1 = model.block1.ca(x)
        sa1 = model.block1.sa(x)
        attention_map1 = (ca1 * sa1).squeeze().cpu().numpy().mean(axis=0)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(spectrogram.squeeze().cpu(), aspect='auto', origin='lower', cmap='magma')
    plt.title('Original Spectrogram')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(attention_map1, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Attention Map (Block 1)')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def multilabel_undersample(X, y, max_per_label=1000, random_state=42):
    np.random.seed(random_state)
    selected_indices = set()

    for i in range(y.shape[1]):
        label_indices = np.where(y[:, i] == 1)[0]
        np.random.shuffle(label_indices)
        selected_indices.update(label_indices[:max_per_label])

    selected_indices = sorted(list(selected_indices))
    return X[selected_indices], y[selected_indices]


def run_experiment(
    num_classes, 
    train_loader, 
    val_loader, 
    test_loader, 
    y_train, 
    mlb, 
    num_epochs=30, 
    lr=1e-3, 
    w_d=1e-2,
    use_attention=True, 
    loss_fn="focal", 
    use_scheduler=False,
    device="mps",
    save_path=None
):
    model = SpectrogramAttentionCNN(num_classes=num_classes, use_attention=use_attention)
    
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        y_train=y_train,
        num_epochs=num_epochs,
        lr=lr,
        loss_fn=loss_fn,
        use_scheduler=use_scheduler,
        device=device
    )

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    y_true, y_pred = evaluate_model(model, test_loader, pd.Series(mlb).to_frame('classes_'), device=device, verbose=0)

    return {
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
    }


def load_model(path, num_classes, device='mps', use_attention=True):
    model = SpectrogramAttentionCNN(num_classes=num_classes, use_attention=use_attention)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def bootstrap_f1(model, test_loader, device='mps', n_iter=100):
    all_images, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            all_images.append(x)
            all_labels.append(y)
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    scores = []
    for _ in range(n_iter):
        idxs = torch.randint(0, len(all_images), (len(all_images),))
        x_bs = all_images[idxs].to(device)
        y_bs = all_labels[idxs].to(device)
        with torch.no_grad():
            preds = (torch.sigmoid(model(x_bs)) > 0.5).float()
        f1 = f1_score(y_bs.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        scores.append(f1)
    return np.array(scores)


