import os
import subprocess
import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset


SEGMENT_DURATION = 4
OVERLAP_DURATION = 2
N_FFT = 2048
HOP_LENGTH = 1536
N_MELS = 128
EPS = 1e-10

FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"


class SpectrogramDataset(Dataset):
    def __init__(self, file_list, labels=[], img_size=(128, 128), augment=False):
        """
        Args:
            file_list (list): List of spectrogram filenames.
            labels (list): Corresponding labels.
            augment (bool): Whether to apply data augmentation.
        """
        self.file_list = file_list
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.augment = augment

        self.base_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype=torch.float32),
                transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
                transforms.Lambda(SpectrogramDataset.normalize),
            ]
        )

        # Data Augmentation
        self.augment_transform = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=(0, 1.0),
                            contrast=(0, 1.0),
                            saturation=(0, 1.0),
                            hue=(0, 0.5),
                        )
                    ],
                    p=0.8,
                ),  # Brightness variation
                transforms.RandomErasing(scale=(0.02, 0.5), p=0.8),  # Random masking
                transforms.Lambda(
                    SpectrogramDataset.add_gaussian_noise
                ),  # Use a static method instead of lambda
                transforms.Lambda(SpectrogramDataset.normalize),
            ]
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        spectrogram, label = load_spectrogram_from_file(self.file_list[idx])
        spectrogram = self.base_transform(spectrogram)

        if self.augment:
            spectrogram = self.augment_transform(spectrogram)

        if len(self.labels):
            label = self.labels[idx]
        return spectrogram, label

    @staticmethod
    def normalize(img):
        """Applies normalization to the spectrogram."""
        return (img - torch.mean(img)) / torch.std(img)

    @staticmethod
    def add_gaussian_noise(img):
        """Applies random Gaussian noise to the spectrogram."""
        noise = torch.randn_like(img) * torch.std(img) * 0.005
        return img + noise


def convert_mp3_to_wav(mp3_path, wav_path):
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    command = [FFMPEG_PATH, "-y", "-i", mp3_path, wav_path]
    subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )


def process_audio_to_melspectrogram(wav_path):
    try:
        data, sr = sf.read(wav_path, dtype=np.int16)
    except Exception as e:
        print(f"Error reading {wav_path}: {e}")
        return []

    if len(data.shape) > 1:
        data = data[:, 0]  # mono

    segment_samples = SEGMENT_DURATION * sr
    overlap_samples = OVERLAP_DURATION * sr
    step_size = segment_samples - overlap_samples

    mel_filters = mel_filter_bank(N_MELS, N_FFT, sr)
    mel_specs = []

    for start in range(0, len(data) - segment_samples + 1, step_size):
        segment = data[start : start + segment_samples]
        f, t, Zxx = scipy.signal.stft(
            segment, fs=sr, nperseg=N_FFT, noverlap=HOP_LENGTH
        )
        magnitude = np.abs(Zxx) ** 2
        mel_spec = np.dot(mel_filters, magnitude)
        mel_spec_db = 10 * np.log10(np.maximum(mel_spec, EPS))
        mel_specs.append(mel_spec_db)

    return mel_specs


def load_spectrogram_from_file(filename):
    """Loads the spectrogram from a .npz file."""
    data = np.load(filename, allow_pickle=True)
    spectrogram = data["spectrogram"]
    label = data["label"]

    return spectrogram, str(label)


def mel_filter_bank(n_mels, n_fft, sr):
    """Generate a Mel filter bank matrix."""
    min_hz, max_hz = 0, sr // 2
    mel_points = np.linspace(hz_to_mel(min_hz), hz_to_mel(max_hz), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        filters[i - 1, bin_points[i - 1] : bin_points[i]] = np.linspace(
            0, 1, bin_points[i] - bin_points[i - 1]
        )
        filters[i - 1, bin_points[i] : bin_points[i + 1]] = np.linspace(
            1, 0, bin_points[i + 1] - bin_points[i]
        )

    return filters


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)
