import os
import torch
from torch.optim import Adam
from tqdm import tqdm
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import sys
sys.path.append("/project/baby-cry-diffusion/src")
from dataset import get_dataloader, N_SAMPLES

DATA_DIR   = "/project/baby-cry-diffusion/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data"
OUT_DIR    = "/project/baby-cry-diffusion/outputs/baseline"
BATCH_SIZE = 8
LR         = 1e-4
EPOCHS     = 50
SAVE_EVERY = 10
GEN_EVERY  = 10
NUM_STEPS  = 50
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

model = DiffusionModel(
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

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

loader = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE)
optimizer = Adam(model.parameters(), lr=LR)

def generate_samples(epoch, n=4):
    model.eval()
    with torch.no_grad():
        noise = torch.randn(n, 1, N_SAMPLES).to(DEVICE)
        samples = model.sample(noise, num_steps=NUM_STEPS)
    import soundfile as sf
    import numpy as np
    for i, s in enumerate(samples):
        path = os.path.join(OUT_DIR, f"epoch{epoch}_sample{i}.wav")
        audio = s.squeeze(0).cpu().numpy()
        sf.write(path, audio, samplerate=16000)
    print(f"Saved {n} samples to {OUT_DIR}")
    model.train()

def train():
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x, _ = batch
            x = x.to(DEVICE)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        if epoch % SAVE_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"baseline_epoch{epoch}.pt"))
        if epoch % GEN_EVERY == 0:
            generate_samples(epoch)

if __name__ == "__main__":
    train()