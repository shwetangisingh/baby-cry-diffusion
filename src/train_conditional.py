import os
import sys
import torch
import torch.nn as nn
import soundfile as sf
from torch.optim import Adam
from tqdm import tqdm
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
sys.path.append("/project/baby-cry-diffusion/src")
from dataset import get_dataloader, get_balanced_dataloader, N_SAMPLES, CLASSES, CLASS2IDX

DATA_DIR    = "/project/baby-cry-diffusion/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data"
OUT_DIR = "/project/baby-cry-diffusion/outputs/conditional_balanced"
BATCH_SIZE  = 8
LR          = 1e-4
EPOCHS      = 100
SAVE_EVERY  = 10
GEN_EVERY   = 10
NUM_STEPS   = 50
NUM_CLASSES = len(CLASSES)
EMBED_DIM   = 128
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# Class embedding - learns a vector for each cry type
class_embedding = nn.Embedding(NUM_CLASSES, EMBED_DIM).to(DEVICE)

model = DiffusionModel(
    net_t=UNetV0,
    in_channels=1,
    channels=[32, 64, 128, 256],
    factors=[4, 4, 4, 4],
    items=[2, 2, 2, 2],
    attentions=[0, 0, 1, 1],
    attention_heads=4,
    attention_features=32,
    embedding_features=EMBED_DIM,
    diffusion_t=VDiffusion,
    sampler_t=VSampler,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in class_embedding.parameters())
print(f"Model parameters: {total_params:,}")

from dataset import get_balanced_dataloader
loader = get_balanced_dataloader(DATA_DIR, batch_size=BATCH_SIZE)
optimizer = Adam(list(model.parameters()) + list(class_embedding.parameters()), lr=LR)

def generate_per_class(epoch, n=2):
    model.eval()
    class_embedding.eval()
    with torch.no_grad():
        for cls_name, cls_idx in CLASS2IDX.items():
            y = torch.tensor([cls_idx] * n).to(DEVICE)
            emb = class_embedding(y).unsqueeze(1)
            noise = torch.randn(n, 1, N_SAMPLES).to(DEVICE)
            samples = model.sample(noise, embedding=emb, num_steps=NUM_STEPS)
            for i, s in enumerate(samples):
                path = os.path.join(OUT_DIR, f"epoch{epoch}_{cls_name}_{i}.wav")
                audio = s.squeeze(0).cpu().numpy()
                sf.write(path, audio, samplerate=16000)
    print(f"Generated samples for all classes at epoch {epoch}")
    model.train()
    class_embedding.train()

def train():
    model.train()
    class_embedding.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            emb = class_embedding(y).unsqueeze(1)
            optimizer.zero_grad()
            loss = model(x, embedding=emb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        if epoch % SAVE_EVERY == 0:
            torch.save({
                'model': model.state_dict(),
                'embedding': class_embedding.state_dict()
            }, os.path.join(OUT_DIR, f"conditional_epoch{epoch}.pt"))
            print(f"Saved checkpoint at epoch {epoch}")
        if epoch % GEN_EVERY == 0:
            generate_per_class(epoch)

if __name__ == "__main__":
    train()
    print("Conditional training complete!")