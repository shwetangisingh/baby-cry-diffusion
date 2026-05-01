import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

SR = 16000
DURATION = 3.0
N_SAMPLES = 49152

CLASSES = ["belly_pain", "discomfort", "hungry", "tired"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}

def load_audio(path):
    wav, _ = librosa.load(path, sr=SR, mono=True)
    return wav

def chunk_audio(wav):
    chunks = []
    if len(wav) < N_SAMPLES:
        pad = __import__("numpy").zeros(N_SAMPLES)
        pad[:len(wav)] = wav
        chunks.append(pad)
    else:
        for start in range(0, len(wav) - N_SAMPLES + 1, N_SAMPLES):
            chunks.append(wav[start:start + N_SAMPLES])
    return chunks

class BabyCryDataset(Dataset):
    def __init__(self, data_dir, classes=CLASSES):
        self.samples = []
        for cls in classes:
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.exists(cls_dir):
                print(f"Warning: {cls_dir} not found, skipping")
                continue
            files = [f for f in os.listdir(cls_dir) if f.endswith(".wav")]
            print(f"Loading {cls}: {len(files)} files")
            for fname in files:
                path = os.path.join(cls_dir, fname)
                try:
                    wav = load_audio(path)
                    chunks = chunk_audio(wav)
                    for chunk in chunks:
                        x = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
                        self.samples.append((x, CLASS2IDX[cls]))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_dataloader(data_dir, batch_size=16, shuffle=True):
    ds = BabyCryDataset(data_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)

if __name__ == "__main__":
    data_dir = "/project/baby-cry-diffusion/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data"
    ds = BabyCryDataset(data_dir)
    if len(ds) > 0:
        x, y = ds[0]
        print(f"Sample shape: {x.shape}, class: {CLASSES[y]}")
