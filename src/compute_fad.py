import os
import sys
import torch
import numpy as np
import soundfile as sf
from frechet_audio_distance import FrechetAudioDistance
sys.path.append("/project/baby-cry-diffusion/src")
from dataset import N_SAMPLES, CLASSES, CLASS2IDX
from train_baseline import model as baseline_model
from train_conditional import model as cond_model, class_embedding

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "/project/baby-cry-diffusion/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data"
OUT_DIR  = "/project/baby-cry-diffusion/outputs/fad_temp"
os.makedirs(f"{OUT_DIR}/real", exist_ok=True)
os.makedirs(f"{OUT_DIR}/baseline", exist_ok=True)
os.makedirs(f"{OUT_DIR}/conditional", exist_ok=True)
NUM_STEPS = 50
N_GEN = 20  # generate 20 samples per model

fad = FrechetAudioDistance(use_pca=False, use_activation=False, verbose=True)

# ── Step 1: Save real audio samples ──────────────────────────────────────────
print("Saving real audio samples...")
count = 0
for cls in CLASSES:
    cls_dir = os.path.join(DATA_DIR, cls)
    for fname in os.listdir(cls_dir):
        if fname.endswith('.wav') and count < 50:
            import librosa
            wav, _ = librosa.load(os.path.join(cls_dir, fname), sr=16000, mono=True, duration=3.0)
            if len(wav) < N_SAMPLES:
                wav = np.pad(wav, (0, N_SAMPLES - len(wav)))
            sf.write(f"{OUT_DIR}/real/real_{count}.wav", wav[:N_SAMPLES], 16000)
            count += 1

print(f"Saved {count} real samples")

# ── Step 2: Generate baseline samples ────────────────────────────────────────
print("\nGenerating baseline samples...")
ckpt = torch.load("outputs/baseline/baseline_epoch50.pt")
baseline_model.load_state_dict(ckpt)
baseline_model.eval()
with torch.no_grad():
    noise = torch.randn(N_GEN, 1, N_SAMPLES).to(DEVICE)
    samples = baseline_model.sample(noise, num_steps=NUM_STEPS)
for i, s in enumerate(samples):
    sf.write(f"{OUT_DIR}/baseline/gen_{i}.wav", s.squeeze(0).cpu().numpy(), 16000)
print(f"Saved {N_GEN} baseline samples")

# ── Step 3: Generate conditional samples ─────────────────────────────────────
print("\nGenerating conditional samples...")
ckpt2 = torch.load("outputs/conditional_balanced/conditional_epoch100.pt")
cond_model.load_state_dict(ckpt2['model'])
class_embedding.load_state_dict(ckpt2['embedding'])
cond_model.eval()
class_embedding.eval()
with torch.no_grad():
    # generate equal samples per class
    all_samples = []
    for cls_name, cls_idx in CLASS2IDX.items():
        y = torch.tensor([cls_idx] * 5).to(DEVICE)
        emb = class_embedding(y).unsqueeze(1)
        noise = torch.randn(5, 1, N_SAMPLES).to(DEVICE)
        s = cond_model.sample(noise, embedding=emb, num_steps=NUM_STEPS)
        all_samples.append(s)
    all_samples = torch.cat(all_samples, dim=0)
for i, s in enumerate(all_samples):
    sf.write(f"{OUT_DIR}/conditional/gen_{i}.wav", s.squeeze(0).cpu().numpy(), 16000)
print(f"Saved {len(all_samples)} conditional samples")

# ── Step 4: Compute FAD ───────────────────────────────────────────────────────
print("\nComputing FAD scores...")
fad_baseline = fad.score(f"{OUT_DIR}/real", f"{OUT_DIR}/baseline")
print(f"FAD (Baseline vs Real):     {fad_baseline:.4f}")

fad_conditional = fad.score(f"{OUT_DIR}/real", f"{OUT_DIR}/conditional")
print(f"FAD (Conditional vs Real):  {fad_conditional:.4f}")

print("\nDone! Lower FAD = more realistic audio.")