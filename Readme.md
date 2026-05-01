# Baby Cry Generation with Audio Diffusion

**CSE 546 Final Project — University at Buffalo, Spring 2026**  
**Team:** Shwetangi Singh, Jeff Zheng

---

## Project Overview

This project explores **conditional audio generation of infant cries** using diffusion models. We fine-tune the `audio-diffusion-pytorch` library on the DonateACry dataset and extend it with class conditioning to generate cry audio for specific cry categories (hungry, discomfort, belly pain, tired).

The project follows the course structure:
1. **Baseline** — Unconditional audio diffusion model trained on all cry categories
2. **Extension** — Class-conditional generation using learned category embeddings
3. **Evaluation** — Classifier-based evaluation of conditional generation quality

---

## Dataset

We use the **DonateACry Corpus** — an open-source dataset of labeled infant cry recordings.

| Class | Files | Chunks (3s) |
|---|---|---|
| hungry | 382 | ~700 |
| discomfort | 27 | ~50 |
| tired | 24 | ~45 |
| belly_pain | 16 | ~30 |
| **Total** | **449** | **898** |

**Preprocessing:** All audio resampled to 16kHz mono, split into 3-second chunks (49152 samples), padded if shorter.

---

## Model Architecture

We use the `audio-diffusion-pytorch` library which operates directly on raw waveforms via a 1D U-Net.