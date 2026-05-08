import os
import sys
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
sys.path.append("/project/baby-cry-diffusion/src")
from dataset import BabyCryDataset, CLASSES, CLASS2IDX, N_SAMPLES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PER_CLASS_DIR = "/project/baby-cry-diffusion/outputs/per_class"
DATA_DIR = "/project/baby-cry-diffusion/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data"

class CryCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, hop_length=128, n_mels=64
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, 128), nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        m = self.mel(x)
        m = torch.log(m + 1e-8)
        if m.dim() == 3:
            m = m.unsqueeze(1)
        return self.fc(self.cnn(m))

def build_model():
    return DiffusionModel(
        net_t=UNetV0,
        in_channels=1,
        channels=[32, 64, 128, 256],
        factors=[4, 4, 4, 4],
        items=[2, 2, 2, 2],
        attentions=[0, 0, 1, 1],
        attention_heads=4,
        attention_features=32,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
    ).to(DEVICE)

def train_classifier():
    print("Training classifier on real data...")
    ds = BabyCryDataset(DATA_DIR)
    n_val = int(0.2 * len(ds))
    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])

    # Balanced sampler - forces equal class representation per batch
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler
    labels = [ds.samples[i][1] for i in train_ds.indices]
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=16)

    clf = CryCNN().to(DEVICE)
    opt = Adam(clf.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(1, 51):  # more epochs too
        clf.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = criterion(clf(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        clf.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = clf(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(clf.state_dict(), "outputs/best_classifier.pt")
        if epoch % 10 == 0:
            print(f"  Epoch {epoch} | Val Acc: {acc:.3f}")
    print(f"Best classifier val accuracy: {best_acc:.3f}")
    clf.load_state_dict(torch.load("outputs/best_classifier.pt"))
    return clf

def evaluate_per_class(clf, n=20):
    print("\nEvaluating per-class models...")
    clf.eval()
    results = {}

    for cls_name, cls_idx in CLASS2IDX.items():
        # Load the per-class model
        model = build_model()
        ckpt = os.path.join(PER_CLASS_DIR, f"{cls_name}_model.pt")
        model.load_state_dict(torch.load(ckpt))
        model.eval()

        with torch.no_grad():
            noise = torch.randn(n, 1, N_SAMPLES).to(DEVICE)
            samples = model.sample(noise, num_steps=50)
            y_true = torch.tensor([cls_idx] * n).to(DEVICE)
            preds = clf(samples).argmax(dim=1)
            acc = (preds == y_true).float().mean().item()

        results[cls_name] = acc
        print(f"  {cls_name}: {acc:.3f}")

    overall = np.mean(list(results.values()))
    print(f"\nOverall per-class accuracy: {overall:.3f}")
    return results

if __name__ == "__main__":
    clf = train_classifier()
    results = evaluate_per_class(clf)