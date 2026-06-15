# outpaint — Image Field-of-View Expander

A diffusion model that seamlessly extends any photo beyond its original borders.
Fine-tuned from [Stable Diffusion 2 Inpainting](https://huggingface.co/sd2-community/stable-diffusion-2-inpainting) on COCO 2017, served through a real-time web interface with a live animated preview.

> Built as a learning project to explore latent diffusion models end-to-end — dataset preparation, U-Net fine-tuning, and inference with a custom web UI.

---

## Demo

**Before → After**

| Original | Expanded |
|---|---|
| ![original](examples/result-original.png) | ![expanded](examples/result-expanded.png) |

**Web UI walkthrough**

| Upload & live preview | Mid-generation (step-by-step reveal) |
|---|---|
| ![ui preview](examples/ui-preview.png) | ![generating](examples/ui-generating.png) |

| Final result | Empty state |
|---|---|
| ![ui result](examples/ui-result.png) | ![ui empty](examples/ui-empty.png) |

---

## How it works

1. **Upload** any image and set how many pixels to add on each side (left / right / top / bottom).
2. The **animated canvas** instantly shows a blurred edge-bleed preview of what the expansion covers — no AI needed yet.
3. Hit **Expand** — the fine-tuned U-Net denoises the masked region over 40 DDIM steps, streaming progress and optional step-by-step previews to the browser in real time.
4. Download the result as PNG or record a `.webm` clip of the animated border.

**Architecture at a glance:**

```
Image  →  VAE encode  →  [noisy latent | mask | masked-image latent]
                                        ↓
                              SD2 U-Net (9-ch input)   ←  CLIP text
                                        ↓
                              DDIM scheduler (40 steps)
                                        ↓
                              VAE decode  →  Expanded image
```

- **VAE**: `stabilityai/sd-vae-ft-mse` — frozen, 8× downscale
- **U-Net**: SD2-inpainting backbone, fine-tuned on COCO 2017 outpainting pairs
- **Text steering**: classifier-free guidance (guidance scale > 1.0 activates it)
- **Streaming**: FastAPI + Server-Sent Events → real-time progress bar and previews
- **VRAM**: runs on 4 GB (RTX 3050 Laptop) with attention slicing + VAE tiling

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/<your-username>/outpaint.git
cd outpaint
```

### 2. Install dependencies

```bash
# CPU-only (no GPU)
pip install -r requirements.txt

# With CUDA (recommended — much faster)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 3. Download the fine-tuned checkpoint

The model weights are hosted on HuggingFace. Download and place them at `checkpoints/unet_final/`:

```bash
# Option A — huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="YassinGhaoui/unet_final",
    local_dir="checkpoints/unet_final"
)
EOF

# Option B — manual
# Download config.json + diffusion_pytorch_model.safetensors from
# https://huggingface.co/YassinGhaoui/unet_final
# and place both files in checkpoints/unet_final/
```

> If no checkpoint is found, the server automatically falls back to the base SD2-inpainting U-Net so you can still run inference.

### 4. Start the server

```bash
python server.py
# Open http://localhost:8000
```



## Training your own checkpoint

The full training pipeline is in [`data_collection.ipynb`](data_collection.ipynb) — designed to run on **Kaggle** (free T4 GPU, 11h session).

**What it does:**
- Loads COCO 2017 train split (118k images) from Kaggle input
- Generates outpainting pairs on-the-fly: random crop → masked canvas → caption
- Fine-tunes the SD2-inpainting U-Net with 8-bit Adam (fits a 15 GB T4)
- Saves rotating checkpoints every 500 steps (keeps last 2 + `unet_latest`) to stay within Kaggle's 20 GB disk quota
- Runs classifier-free guidance training with 10% caption dropout

**Key training settings:**

| Setting | Value |
|---|---|
| Base model | `sd2-community/stable-diffusion-2-inpainting` |
| Dataset | COCO 2017 train (118k images) |
| Batch size | 4 |
| Learning rate | 1e-5 with 200-step warmup |
| Max training | 11h / ~50k steps |
| Optimizer | `bitsandbytes.AdamW8bit` |
| Mixed precision | fp16 |
| Canvas size | 512×512 (crop 384×384 inner) |

---

## Project structure

```
outpaint/
├── server.py               # FastAPI backend — model loading, inference, SSE streaming
├── web/
│   └── index.html          # Single-page UI — animated canvas, controls, live preview
├── data_collection.ipynb   # Kaggle training notebook
├── examples/               # Sample outputs
├── requirements.txt
└── .gitignore
```

---

## Controls

| Control | What it does |
|---|---|
| Left / Right / Top / Bottom sliders | Pixels to add on each side |
| Max output side | Caps the longest canvas dimension (lower = faster, less VRAM) |
| Steps | DDIM denoising steps (20 = fast draft, 40 = default, 60+ = diminishing returns) |
| Prompt | Describe what to generate — only active when guidance > 1.0 |
| Guidance scale | 1.0 = seamless fill (ignores prompt), 7–8 = prompt-directed |
| Seed | Reproducibility |
| Step-by-step preview | Decode a frame every N steps — watch the image emerge from noise |

---

## Requirements

- Python 3.10+
- PyTorch 2.6+ with CUDA (runs on CPU but very slow)
- ~6 GB disk for model weights
- 4+ GB VRAM recommended (runs on RTX 3050 Laptop 4 GB)

---

## License

Model weights inherit the [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL) from Stable Diffusion 2.
Code in this repository is MIT licensed.
