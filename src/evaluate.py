import os
import sys
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
sys.path.append("/project/baby-cry-diffusion/src")
from dataset import BabyCryDataset, CLASSES, CLASS2IDX, N_SAMPLES
from train_conditional import model, class_embedding, NUM_STEPS, DEVICE, EMBED_DIM, NUM_CLASSES

# ── CNN Classifier ────────────────────────────────────────────────────────────
# Takes raw waveform → mel spectrogram → CNN → class prediction
class CryCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, hop_length=128, n_mels=64
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        m = self.mel(x)
        m = torch.log(m + 1e-8)
        if m.dim() == 3:
            m = m.unsqueeze(1)
        return self.fc(self.cnn(m))

# ── Train Classifier on Real Data ─────────────────────────────────────────────
def train_classifier(data_dir, epochs=30):
    print("Training classifier on REAL data...")
    ds = BabyCryDataset(data_dir)
    n_val = int(0.2 * len(ds))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16)

    clf = CryCNN().to(DEVICE)
    opt = Adam(clf.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        clf.train()
        for x, y in tqdm(train_loader, desc=f"Classifier Epoch {epoch}/{epochs}"):
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
        print(f"Epoch {epoch} | Val Accuracy: {correct/total:.3f}")

    torch.save(clf.state_dict(), "outputs/classifier.pt")
    print("Classifier saved!")
    return clf

# ── Evaluate Generated Samples ────────────────────────────────────────────────
def evaluate_generated(clf, n_per_class=20):
    print("\nEvaluating generated samples...")
    clf.eval()
    model.eval()
    class_embedding.eval()
    results = {}

    with torch.no_grad():
        for cls_name, cls_idx in CLASS2IDX.items():
            y = torch.tensor([cls_idx] * n_per_class).to(DEVICE)
            emb = class_embedding(y).unsqueeze(1)
            noise = torch.randn(n_per_class, 1, N_SAMPLES).to(DEVICE)
            samples = model.sample(noise, embedding=emb, num_steps=NUM_STEPS)
            logits = clf(samples)
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()
            results[cls_name] = acc
            print(f"  {cls_name}: classifier accuracy = {acc:.3f}")

    overall = np.mean(list(results.values()))
    print(f"\nOverall classifier accuracy on generated samples: {overall:.3f}")
    return results

if __name__ == "__main__":
    DATA_DIR = "/project/baby-cry-diffusion/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data"
    os.makedirs("outputs", exist_ok=True)

    # Step 1: train classifier on real data
    clf = train_classifier(DATA_DIR)

    # Step 2: load trained conditional model checkpoint
    ckpt = torch.load("outputs/conditional_balanced/conditional_epoch100.pt")
    model.load_state_dict(ckpt['model'])
    class_embedding.load_state_dict(ckpt['embedding'])

    # Step 3: evaluate generated samples
    results = evaluate_generated(clf)