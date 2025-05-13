import os
import subprocess
from tqdm import tqdm

FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"

def convert_mp3_to_wav(mp3_path, wav_path):
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    command = [
        FFMPEG_PATH,
        "-y",
        "-i", mp3_path,
        wav_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def process_ismir_dataset(input_dir, output_dir):
    mp3_files = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(".mp3"):
                continue

            mp3_path = os.path.join(root, file)
            rel_parts = os.path.relpath(mp3_path, input_dir).split(os.sep)

            if len(rel_parts) < 3:
                continue

            genre = rel_parts[1]
            filename = "_".join(rel_parts[2:])
            filename = os.path.splitext(filename)[0] + ".wav"

            wav_path = os.path.join(output_dir, genre, filename)
            mp3_files.append((mp3_path, wav_path))

    for mp3_path, wav_path in tqdm(mp3_files, desc="Конвертация MP3 в WAV", ncols=100, dynamic_ncols=True, leave=False):
        try:
            convert_mp3_to_wav(mp3_path, wav_path)
        except subprocess.CalledProcessError:
            print(f"Ошибка конвертации: {mp3_path}")

    print(f"✔️ Конвертация завершена: {len(mp3_files)} файлов")
