"""
FastAPI backend for the Image Field-of-View Expander web UI.

It serves web/index.html (the live, animated, blurred-edge preview — pure browser,
no model needed) and exposes /api/expand which runs the actual diffusion model.

Run:
    pip install fastapi "uvicorn[standard]" python-multipart torch diffusers transformers accelerate pillow
    python server.py
    # then open http://localhost:8000

Model: loads checkpoints/unet_final if present, else falls back to base SD2-inpainting.
"""

import base64
import io
import math
import time
from pathlib import Path

import torch
import torchvision.transforms as TT
from PIL import Image, ImageFilter
import json
import queue
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_REPO = "sd2-community/stable-diffusion-2-inpainting"
VAE_REPO = "stabilityai/sd-vae-ft-mse"
CHECKPOINT_DIR = Path("checkpoints/unet_final")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Image Field-of-View Expander")

# Loaded lazily on first request so the server starts instantly.
_models = None


def _snap8(x):
    return int(math.ceil(x / 8) * 8)


def load_models():
    global _models
    if _models is not None:
        return _models

    print(f"[load] device={DEVICE} dtype={DTYPE}")
    vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=DTYPE).to(DEVICE).eval()

    if CHECKPOINT_DIR.exists():
        unet = UNet2DConditionModel.from_pretrained(CHECKPOINT_DIR, torch_dtype=DTYPE).to(DEVICE).eval()
        source = f"fine-tuned ({CHECKPOINT_DIR})"
    else:
        unet = UNet2DConditionModel.from_pretrained(
            BASE_REPO, subfolder="unet", torch_dtype=DTYPE).to(DEVICE).eval()
        source = "base SD2-inpainting (no fine-tuned checkpoint found)"

    text_encoder = CLIPTextModel.from_pretrained(
        BASE_REPO, subfolder="text_encoder", torch_dtype=DTYPE).to(DEVICE).eval()
    tokenizer = CLIPTokenizer.from_pretrained(BASE_REPO, subfolder="tokenizer")

    scheduler = DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False, prediction_type="epsilon")

    # Low-VRAM measures (essential on a 4 GB GPU).
    if DEVICE == "cuda":
        vae.enable_slicing()
        vae.enable_tiling()
        try:
            unet.set_attention_slice("auto")
        except Exception:
            pass
        torch.cuda.empty_cache()

    print(f"[load] model ready: {source}")
    _models = (vae, unet, text_encoder, tokenizer, scheduler, source)
    return _models


@torch.no_grad()
def _encode_prompt(text_encoder, tokenizer, prompt):
    tok = tokenizer([prompt], padding="max_length", truncation=True,
                    max_length=tokenizer.model_max_length, return_tensors="pt").to(DEVICE)
    return text_encoder(tok.input_ids)[0]


@torch.no_grad()
def _latents_to_pil(vae, latents):
    """Decode a latent tensor to a PIL image (used for both the final result and
    the optional step-by-step previews)."""
    sf = vae.config.scaling_factor
    img = vae.decode(latents / sf).sample
    img = ((img.clamp(-1, 1) + 1) / 2).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((img * 255).astype("uint8"))


def _composite_original(result, inner, off_x, off_y, expand_px, feather=24):
    """Paste the sharp original pixels (`inner`) back over the generated canvas
    (`result`) at (off_x, off_y), feathering the inner edge so the original blends
    smoothly into the generated border instead of a hard cut.

    The VAE encode→decode cycle subtly shifts the colors/sharpness of the known
    region, so without this the cropped part never matches the input exactly. The
    feather is only applied on sides that were actually expanded — a side with no
    expansion keeps the original right up to the canvas edge (no blending needed).
    """
    import numpy as np
    in_w, in_h = inner.size
    L, R, T, B = expand_px
    f = max(1, min(feather, in_w // 2, in_h // 2))

    # Per-axis ramps: 0 at an expanded edge climbing to 1 over `f` px, else flat 1.
    xr = np.ones(in_w, dtype=np.float32)
    if L > 0:
        xr = np.minimum(xr, np.clip(np.arange(in_w) / f, 0, 1))
    if R > 0:
        xr = np.minimum(xr, np.clip((in_w - 1 - np.arange(in_w)) / f, 0, 1))
    yr = np.ones(in_h, dtype=np.float32)
    if T > 0:
        yr = np.minimum(yr, np.clip(np.arange(in_h) / f, 0, 1))
    if B > 0:
        yr = np.minimum(yr, np.clip((in_h - 1 - np.arange(in_h)) / f, 0, 1))

    alpha_arr = (np.outer(yr, xr) * 255).astype("uint8")
    alpha = Image.fromarray(alpha_arr, "L").filter(ImageFilter.GaussianBlur(f / 3))

    out = result.convert("RGB").copy()
    out.paste(inner, (off_x, off_y), alpha)
    return out


@torch.no_grad()
def run_outpaint(image, expand, max_side, steps, prompt, guidance_scale, seed=0,
                 on_step=None, preview_every=0, on_preview=None):
    """on_step(i, total): fired after each denoising step (drives the progress bar).
    preview_every: if >0, decode + emit an intermediate image every N steps via
                   on_preview(i, total, pil_image) — the 'emerging from noise' reveal.
                   Costs one extra VAE decode each time, so keep N reasonable on a
                   small GPU (0 = off)."""
    vae, unet, text_encoder, tokenizer, scheduler, _ = load_models()
    image = image.convert("RGB")
    L, R, T, B = expand
    iw, ih = image.size
    cw, ch = iw + L + R, ih + T + B

    scale = min(1.0, max_side / max(cw, ch))
    out_w, out_h = _snap8(cw * scale), _snap8(ch * scale)
    in_w, in_h = max(8, _snap8(iw * scale)), max(8, _snap8(ih * scale))
    off_x = min(int(round(L * scale)), out_w - in_w)
    off_y = min(int(round(T * scale)), out_h - in_h)

    inner = image.resize((in_w, in_h), Image.LANCZOS)
    to_t = TT.Compose([TT.ToTensor(), TT.Normalize([0.5] * 3, [0.5] * 3)])
    inner_t = to_t(inner).unsqueeze(0).to(DEVICE, DTYPE)

    masked_image = torch.zeros(1, 3, out_h, out_w, device=DEVICE, dtype=DTYPE)
    masked_image[:, :, off_y:off_y + in_h, off_x:off_x + in_w] = inner_t
    mask = torch.zeros(1, 1, out_h, out_w, device=DEVICE, dtype=DTYPE)
    mask[:, :, off_y:off_y + in_h, off_x:off_x + in_w] = 1.0

    g = torch.Generator(device=DEVICE).manual_seed(int(seed))
    latents = torch.randn(1, 4, out_h // 8, out_w // 8, device=DEVICE, dtype=DTYPE, generator=g)

    use_cfg = guidance_scale and guidance_scale > 1.0
    enc_cond = _encode_prompt(text_encoder, tokenizer, prompt)
    enc_uncond = _encode_prompt(text_encoder, tokenizer, "") if use_cfg else None

    scheduler.set_timesteps(steps)
    sf = vae.config.scaling_factor
    masked_latent = vae.encode(masked_image).latent_dist.sample() * sf
    mask_lat = torch.nn.functional.interpolate(mask, size=latents.shape[-2:], mode="nearest")

    total = len(scheduler.timesteps)
    for i, t in enumerate(scheduler.timesteps):
        unet_input = torch.cat([latents, mask_lat, masked_latent], dim=1)
        t_in = t.unsqueeze(0).to(DEVICE)
        if use_cfg:
            inp2 = torch.cat([unet_input, unet_input], dim=0)
            enc2 = torch.cat([enc_uncond, enc_cond], dim=0)
            n_un, n_co = unet(inp2, t_in.expand(2), encoder_hidden_states=enc2).sample.chunk(2)
            noise_pred = n_un + guidance_scale * (n_co - n_un)
        else:
            noise_pred = unet(unet_input, t_in, encoder_hidden_states=enc_cond).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        if on_step is not None:
            on_step(i + 1, total)

        # Optional step-by-step reveal: decode the current latent into a viewable
        # frame every `preview_every` steps. Skip the very last step (the final
        # full-quality decode happens below anyway).
        if (preview_every and on_preview is not None
                and (i + 1) % preview_every == 0 and (i + 1) < total):
            try:
                on_preview(i + 1, total, _latents_to_pil(vae, latents))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()  # don't kill the run for a preview frame

    result = _latents_to_pil(vae, latents)
    # Paste the sharp original back over the known region with a feathered edge, so
    # that part matches the input exactly and blends seamlessly into the new border.
    return _composite_original(result, inner, off_x, off_y, (L, R, T, B))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
class ExpandRequest(BaseModel):
    image: str            # data URL (data:image/...;base64,...)
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0
    max_side: int = 512
    steps: int = 40
    prompt: str = ""
    guidance_scale: float = 1.0
    seed: int = 0
    preview_every: int = 0   # 0 = off; >0 = stream a decoded frame every N steps


def _dataurl_to_image(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data_url)))


def _image_to_dataurl(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    return {"device": DEVICE, "checkpoint": str(CHECKPOINT_DIR), "loaded": _models is not None}


@app.post("/api/expand")
def expand(req: ExpandRequest):
    try:
        image = _dataurl_to_image(req.image)
        result = run_outpaint(
            image,
            expand=(req.left, req.right, req.top, req.bottom),
            max_side=req.max_side, steps=req.steps,
            prompt=req.prompt, guidance_scale=req.guidance_scale, seed=req.seed,
        )
        return {"image": _image_to_dataurl(result),
                "size": list(result.size)}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507,
                            detail="GPU out of memory — lower 'Max output side' or 'Steps'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/expand_stream")
def expand_stream(req: ExpandRequest):
    """Server-Sent Events: streams a 'progress' event per denoising step, then a
    final 'done' event with the image. Lets the browser show a real progress bar."""
    q: "queue.Queue" = queue.Queue()

    def worker():
        try:
            image = _dataurl_to_image(req.image)

            def on_step(i, total):
                q.put({"type": "progress", "step": i, "total": total})

            def on_preview(i, total, pil_img):
                q.put({"type": "preview", "step": i, "total": total,
                       "image": _image_to_dataurl(pil_img)})

            t0 = time.perf_counter()
            result = run_outpaint(
                image,
                expand=(req.left, req.right, req.top, req.bottom),
                max_side=req.max_side, steps=req.steps,
                prompt=req.prompt, guidance_scale=req.guidance_scale, seed=req.seed,
                on_step=on_step,
                preview_every=req.preview_every, on_preview=on_preview,
            )
            if DEVICE == "cuda":
                torch.cuda.synchronize()  # CUDA is async — wait for real completion
            elapsed = time.perf_counter() - t0
            per_step = elapsed / max(1, req.steps)
            print(f"[bench] {result.size[0]}x{result.size[1]} · {req.steps} steps · "
                  f"{elapsed:.2f}s total · {per_step:.3f}s/step")
            q.put({"type": "done", "image": _image_to_dataurl(result),
                   "size": list(result.size),
                   "elapsed": round(elapsed, 2),
                   "per_step": round(per_step, 3)})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            q.put({"type": "error", "detail": "GPU out of memory — lower 'Max output side' or 'Steps'."})
        except Exception as e:
            q.put({"type": "error", "detail": str(e)})
        finally:
            q.put(None)  # sentinel: stream finished

    threading.Thread(target=worker, daemon=True).start()

    def event_stream():
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    print("Open http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
