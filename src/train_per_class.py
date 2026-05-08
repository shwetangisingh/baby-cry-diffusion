import os
import sys
import torch
import soundfile as sf
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
sys.path.append("/project/baby-cry-diffusion/src")
from dataset import BabyCryDataset, CLASSES, N_SAMPLES

DATA_DIR  = "/project/baby-cry-diffusion/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data"
OUT_DIR   = "/project/baby-cry-diffusion/outputs/per_class"
BATCH_SIZE = 4
LR        = 1e-4
EPOCHS    = 150
NUM_STEPS = 50
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

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

def train_one_class(cls_name):
    print(f"\n{'='*40}")
    print(f"Training model for class: {cls_name}")
    print(f"{'='*40}")

    # Load only this class
    ds = BabyCryDataset(DATA_DIR, classes=[cls_name])
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print(f"Samples: {len(ds)}")

    model = build_model()
    optimizer = Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for x, _ in loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        if epoch % 50 == 0:
            print(f"  Epoch {epoch} | Loss: {avg:.4f}")

    # Save model
    ckpt_path = os.path.join(OUT_DIR, f"{cls_name}_model.pt")
    torch.save(model.state_dict(), ckpt_path)

    # Generate samples
    model.eval()
    with torch.no_grad():
        noise = torch.randn(4, 1, N_SAMPLES).to(DEVICE)
        samples = model.sample(noise, num_steps=NUM_STEPS)
    for i, s in enumerate(samples):
        path = os.path.join(OUT_DIR, f"{cls_name}_gen_{i}.wav")
        sf.write(path, s.squeeze(0).cpu().numpy(), samplerate=16000)
    print(f"  Generated 4 samples for {cls_name}")
    return model

if __name__ == "__main__":
    for cls in CLASSES:
        train_one_class(cls)
    print("\nAll per-class models trained!")