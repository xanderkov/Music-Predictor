import os
import pickle
import numpy as np
from datasets import load_dataset
import scipy.signal
from glob import glob
import multiprocessing
import soundfile as sf
from tqdm import tqdm
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from PIL import Image

SEGMENT_DURATION = 4
OVERLAP_DURATION = 2
N_FFT=2048
HOP_LENGTH=1536
N_MELS=128
EPS = 1e-10
SPECTROGRAM_DIR = "spectrogram_cache"
IMAGE_CACHE_DIR = "image_cache"

os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)


def load_audio_data(data_dir):
    """
    Load audio files from a directory structure where each subfolder represents a genre.
    """
    genres = os.listdir(data_dir)
    audio_files = []
    labels = []

    for genre in genres:
        genre_path = os.path.join(data_dir, genre)
        if os.path.isdir(genre_path):
            files = glob(os.path.join(genre_path, "*.wav"))
            for file in files:
                audio_files.append(file)
                labels.append(genre)

    return audio_files, labels


def load_music_tags_dataset():
    """
    Load the 'luli0034/music-tags-to-spectrogram' dataset.
    """
    dataset = load_dataset("luli0034/music-tags-to-spectrogram")
    return dataset


def save_spectrogram_to_file(spectrogram, label):
    """Saves the spectrogram to a compressed .npz file and returns the filename."""
    filename = f"{SPECTROGRAM_DIR}/{hash(label)}_{np.random.randint(100000)}.npz"
    
    np.savez_compressed(filename, spectrogram=spectrogram, label=label)
    
    return filename, label


def load_spectrogram_from_file(filename):
    """Loads the spectrogram from a .npz file."""
    data = np.load(filename, allow_pickle=True)
    spectrogram = data['spectrogram']
    label = data['label']
    
    return spectrogram, str(label)


def save_image(image_array, label):
    """Save cropped image as a compressed .npz file and return the filename."""
    filename = f"{IMAGE_CACHE_DIR}/{hash(label)}_{np.random.randint(100000)}.npz"
    
    np.savez_compressed(filename, spectrogram=image_array, label=label)
    
    return filename, label

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

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(dtype=torch.float32),
            transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
            transforms.Lambda(SpectrogramDataset.normalize),
        ])

        # Data Augmentation
        self.augment_transform = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=(0, 1.0), 
                                  contrast=(0, 1.0), 
                                  saturation=(0, 1.0), 
                                  hue=(0, 0.5),
                                                          )
                                   ], p=0.8),  # Brightness variation
            transforms.RandomErasing(scale=(0.02, 0.5), p=0.8),  # Random masking
            transforms.Lambda(SpectrogramDataset.add_gaussian_noise),  # Use a static method instead of lambda
            transforms.Lambda(SpectrogramDataset.normalize),
        ])

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


def sample_image(image, label):
    """
    Splits a wide spectrogram image into smaller cropped segments and saves them.
    Returns a list of filenames instead of keeping images in memory.
    """
    image = image.convert("L")
    width, height = image.size
    left = 0
    IMAGE_SIZE = height
    right = 2 * IMAGE_SIZE
    filenames = []

    while right <= width - IMAGE_SIZE:
        cropped_image = image.crop((left, 0, right, height))
        filenames.append(save_image(255.0 - np.array(cropped_image), label))
        left = right
        right = left + IMAGE_SIZE

    return filenames


def _process_single_image(item):
    """Helper function to process a single image (for multiprocessing)."""
    return sample_image(item['image'], item['text'])


def process_dataset(dataset, n_processes=4):
    """Processes dataset images in parallel and saves cropped PNG files."""

    with multiprocessing.Pool(n_processes) as pool:
        results = []
        
        with tqdm(total=len(dataset), desc="Preparing images", unit="file") as pbar:
            for filenames in pool.imap_unordered(_process_single_image, dataset):
                results.extend(filenames)
                pbar.update(1)

    return results


def convert_spectrogram_image_to_mel(spectrogram_img, n_mels=128, sr=26000, n_fft=10_000):
    """Convert a spectrogram image to a Mel spectrogram and save to file."""
    spectrogram_img, _ = spectrogram_img
    spectrogram_img, label = load_spectrogram_from_file(spectrogram_img)
    spectrogram_img = spectrogram_img.astype(np.float32) / 255.0
    spectrogram_img = np.flipud(spectrogram_img)

    mel_filters = mel_filter_bank(n_mels, n_fft, sr)
    if spectrogram_img.shape[0] != mel_filters.shape[1]:
        spectrogram_img = cv2.resize(spectrogram_img, (spectrogram_img.shape[1], mel_filters.shape[1]))

    mel_spectrogram = np.dot(mel_filters, spectrogram_img)
    mel_spectrogram_db = 10 * np.log10(np.maximum(mel_spectrogram, EPS))

    return save_spectrogram_to_file(mel_spectrogram_db, label)


def process_images_in_parallel(dataset, n_processes=4):
    """Process dataset images into spectrograms in parallel and save files."""
    images = process_dataset(dataset, n_processes)

    with multiprocessing.Pool(n_processes) as pool:
        file_list = []
        with tqdm(total=len(images), desc="Processing images", unit="file") as pbar:
            for file_name in pool.imap_unordered(convert_spectrogram_image_to_mel, images):
                file_list.append(file_name)
                pbar.update(1)
    return file_list


def process_audio_to_melspectrogram(audio_file, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """Converts an audio file into Mel spectrogram segments (4s chunks with 2s overlap)."""
    audio_file, label = audio_file
    try:
        data, sr = sf.read(audio_file, dtype=np.int16)
    except Exception as e:
        print(f"Error reading {audio_file}: {e}")
        return []

    if len(data.shape) > 1:
        data = data[:, 0]  # Convert stereo to mono

    segment_samples = SEGMENT_DURATION * sr
    overlap_samples = OVERLAP_DURATION * sr
    step_size = segment_samples - overlap_samples

    mel_filters = mel_filter_bank(n_mels, n_fft, sr)
    file_list = []

    for start in range(0, len(data) - segment_samples + 1, step_size):
        segment = data[start: start + segment_samples]

        f, t, Zxx = scipy.signal.stft(segment, fs=sr, nperseg=n_fft, noverlap=hop_length)
        magnitude = np.abs(Zxx) ** 2
        
        mel_spectrogram = np.dot(mel_filters, magnitude)
        mel_spectrogram_db = 10 * np.log10(np.maximum(mel_spectrogram, EPS))

        file_list.append(save_spectrogram_to_file(mel_spectrogram_db, label))

    return file_list

def process_audio_in_parallel(audio_files, labels, n_processes=4):
    assert len(audio_files) == len(labels)
    audio_files = [(af, l) for af, l in zip(audio_files, labels)]
    
    with multiprocessing.Pool(n_processes) as pool:
        file_list = []
        with tqdm(total=len(audio_files), desc="Processing audios", unit="file") as pbar:
            for file_names in pool.imap_unordered(process_audio_to_melspectrogram, audio_files):
                file_list.extend(file_names)
                pbar.update(1)
    
    return file_list


def mel_filter_bank(n_mels, n_fft, sr):
    """Generate a Mel filter bank matrix."""
    min_hz, max_hz = 0, sr // 2
    mel_points = np.linspace(hz_to_mel(min_hz), hz_to_mel(max_hz), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        filters[i - 1, bin_points[i - 1]:bin_points[i]] = np.linspace(0, 1, bin_points[i] - bin_points[i - 1])
        filters[i - 1, bin_points[i]:bin_points[i + 1]] = np.linspace(1, 0, bin_points[i + 1] - bin_points[i])

    return filters


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)
