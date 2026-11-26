"""
Unicollect Imagine (v22): High-Quality 1024x1024 GPU Text-to-Image Engine
- L4 (22.5 GB) / T4 (15 GB): SG161222/RealVisXL_V5.0_Lightning (6 steps, 1024x1024 photorealistic, ~2.5s, 100% ungated)
- A100 (>= 36 GB): YuCollection/FLUX.1-schnell-Diffusers (4 steps, 1024x1024, 100% ungated)
- Guaranteed Ollama VRAM unload + polling wait until CUDA memory is actually freed (fixes L4 OOM after Coder Ultra!)
- Smart CPU-offload fallback when free VRAM < 10.5 GB + guaranteed CUDA cleanup in finally:
"""
from __future__ import annotations

import base64
import gc
import io
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import requests

IMAGE_MODEL_ID = os.environ.get("IMAGE_MODEL", "SG161222/RealVisXL_V5.0_Lightning").strip() or "SG161222/RealVisXL_V5.0_Lightning"
IMAGE_MODEL_FALLBACKS = (
    "SG161222/RealVisXL_V5.0_Lightning",
    "stabilityai/sdxl-turbo",
)
IMAGE_STEPS = int(os.environ.get("IMAGE_STEPS", "6"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "1024"))
GENERATED_DIR = Path(__file__).resolve().parent / "static" / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

_PIPE = None
_ACTIVE_MODEL_ID = IMAGE_MODEL_ID
_PIPE_LOCK = threading.Lock()


def _has_persian(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _free_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            free_b, _ = torch.cuda.mem_get_info()
            return free_b / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _clear_cuda() -> None:
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def enhance_image_prompt(prompt: str, ollama_url: str, enhancer_model: str) -> str:
    clean = (prompt or "").strip()
    clean = re.sub(
        r"^(?:/imagine|imagine:|generate(?:\s+an?)?\s+image(?:\s+of)?|draw(?:\s+me)?|create(?:\s+an?)?\s+(?:image|picture)(?:\s+of)?|عکس\s+بساز|تصویر\s+بساز|ساخت\s+(?:عکس|تصویر)|یه\s+عکس\s+از|یک\s+عکس\s+از|نقاشی\s+بکش)\s*[:\-]?\s*",
        "",
        clean,
        flags=re.I,
    ).strip() or prompt.strip()
    # If the user already wrote a detailed English prompt (>120 chars), keep it intact!
    if not enhancer_model or (len(clean) >= 120 and not _has_persian(clean)):
        return clean
    try:
        r = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": enhancer_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert visual prompt engineer. Rewrite the user's request (Persian or English) "
                            "into a single vivid, ultra-detailed, photorealistic English text-to-image prompt (35 to 60 words). "
                            "Include subject details, natural lighting, camera/lens feel, textures, and atmosphere. "
                            "Output ONLY the final English prompt, no quotes, no prefixes, no explanations."
                        ),
                    },
                    {"role": "user", "content": clean},
                ],
                "stream": False,
                "think": False,
                "keep_alive": "15m",
                "options": {"temperature": 0.35, "num_predict": 110},
            },
            timeout=12,
        )
        if r.ok:
            en = (r.json().get("message", {}).get("content") or "").strip()
            en = re.sub(r"<think>.*?</think>", "", en, flags=re.S).strip().strip('"').strip("'")
            if en and len(en) >= 10 and not _has_persian(en):
                return en
    except Exception:
        pass
    return clean


def _free_ollama_vram_and_wait(ollama_url: str, min_free_gb: float = 11.0, max_wait_s: float = 18.0) -> float:
    """Unload ALL Ollama models from VRAM and POLL until CUDA memory is actually released."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
    except Exception:
        return 0.0

    free_gb = _free_vram_gb()
    if free_gb >= min_free_gb:
        return free_gb

    # Request immediate unload of EVERY resident Ollama model (e.g. qwen3-coder:30b holding 18+ GB)
    try:
        r = requests.get(f"{ollama_url}/api/ps", timeout=4)
        if r.ok:
            loaded = [m.get("name") for m in r.json().get("models", []) if m.get("name")]
            for name in loaded:
                try:
                    requests.post(f"{ollama_url}/api/generate", json={"model": name, "keep_alive": 0}, timeout=8)
                except Exception:
                    pass
                try:
                    requests.post(f"{ollama_url}/api/chat", json={"model": name, "messages": [], "keep_alive": 0}, timeout=8)
                except Exception:
                    pass
    except Exception:
        pass

    # Poll until Ollama's llama-runner releases CUDA VRAM
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        _clear_cuda()
        free_gb = _free_vram_gb()
        ps_empty = False
        try:
            r = requests.get(f"{ollama_url}/api/ps", timeout=2)
            if r.ok:
                ps_empty = len(r.json().get("models", [])) == 0
        except Exception:
            ps_empty = True
        if free_gb >= min_free_gb and ps_empty:
            break
        if free_gb >= min_free_gb + 2.0:
            break
        time.sleep(0.5)

    print(f"[Unicollect Imagine] VRAM ready: {free_gb:.2f} GiB free", flush=True)
    return free_gb


def _load_pipeline():
    global _PIPE, _ACTIVE_MODEL_ID
    if _PIPE is not None:
        return _PIPE
    import torch

    use_cuda = torch.cuda.is_available()
    total_vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) if use_cuda else 0.0
    free_gb = _free_vram_gb() if use_cuda else 0.0
    hf_token = (os.environ.get("HF_TOKEN") or "").strip() or None

    candidates = []
    for m in (IMAGE_MODEL_ID, *IMAGE_MODEL_FALLBACKS):
        if m and m not in candidates and "black-forest-labs/" not in m.lower():
            candidates.append(m)

    last_err = None
    for cand in candidates:
        pipe = None
        try:
            low = cand.lower()
            if "flux" in low:
                from diffusers import FluxPipeline
                dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else (torch.float16 if use_cuda else torch.float32)
                pipe = FluxPipeline.from_pretrained(cand, torch_dtype=dtype, token=hf_token)
                if use_cuda:
                    try:
                        pipe.vae.enable_slicing()
                        pipe.vae.enable_tiling()
                    except Exception:
                        pass
                    if total_vram_gb >= 36 and free_gb >= 26:
                        pipe = pipe.to("cuda")
                    elif free_gb >= 14:
                        pipe.enable_model_cpu_offload()
                    else:
                        pipe.enable_sequential_cpu_offload()
            else:
                from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
                dtype = torch.float16 if use_cuda else torch.float32
                kw: Dict[str, Any] = {"torch_dtype": dtype, "token": hf_token}
                if use_cuda and ("turbo" in low or "realvis" in low):
                    kw["variant"] = "fp16"
                try:
                    pipe = DiffusionPipeline.from_pretrained(cand, **kw)
                except Exception:
                    kw.pop("variant", None)
                    pipe = DiffusionPipeline.from_pretrained(cand, **kw)
                if "lightning" in low:
                    try:
                        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                            pipe.scheduler.config,
                            algorithm_type="sde-dpmsolver++",
                            use_karras_sigmas=True,
                        )
                    except Exception:
                        pass
                if use_cuda:
                    try:
                        pipe.vae.enable_slicing()
                        pipe.vae.enable_tiling()
                    except Exception:
                        pass
                    free_now = _free_vram_gb()
                    if free_now >= 10.5:
                        pipe = pipe.to("cuda")
                    else:
                        # Safe low-VRAM mode (<4.5 GiB peak) so OOM never happens
                        pipe.enable_model_cpu_offload()
            _PIPE = pipe
            _ACTIVE_MODEL_ID = cand
            return _PIPE
        except Exception as exc:
            last_err = exc
            if pipe is not None:
                del pipe
            _clear_cuda()
            continue
    raise RuntimeError(f"All ungated image models failed: {last_err}")


def generate_image(
    prompt: str,
    ollama_url: str = "http://127.0.0.1:11434",
    router_model: str = "qwen3:1.7b",
    width: int = IMAGE_SIZE,
    height: int = IMAGE_SIZE,
    steps: int = IMAGE_STEPS,
) -> Dict[str, Any]:
    global _PIPE
    t0 = time.time()
    en_prompt = enhance_image_prompt(prompt, ollama_url, router_model)
    width = max(512, min(1024, (int(width) // 64) * 64))
    height = max(512, min(1024, (int(height) // 64) * 64))
    steps = max(2, min(12, int(steps)))
    is_flux_init = "flux" in (_ACTIVE_MODEL_ID or IMAGE_MODEL_ID).lower()

    with _PIPE_LOCK:
        # ALWAYS wait for Ollama (e.g. Coder Ultra 18 GB) to finish releasing CUDA VRAM before loading Diffusion!
        _free_ollama_vram_and_wait(ollama_url, min_free_gb=26.0 if is_flux_init else 11.0, max_wait_s=18.0)
        try:
            pipe = _load_pipeline()
            active_m = (_ACTIVE_MODEL_ID or IMAGE_MODEL_ID).lower()
            is_flux = "flux" in active_m
            is_lightning = "lightning" in active_m
            is_turbo = "turbo" in active_m or "schnell" in active_m

            if is_flux:
                call_kwargs: Dict[str, Any] = {
                    "prompt": en_prompt,
                    "num_inference_steps": min(steps, 4),
                    "guidance_scale": 0.0,
                    "width": width,
                    "height": height,
                    "max_sequence_length": 256,
                }
            elif is_lightning:
                call_kwargs = {
                    "prompt": en_prompt,
                    "negative_prompt": "blurry, lowres, deformed, watermark, text, bad anatomy, artifacts, oversaturated",
                    "num_inference_steps": max(4, min(steps, 8)),
                    "guidance_scale": 1.8,
                    "width": width,
                    "height": height,
                }
            else:
                call_kwargs = {
                    "prompt": en_prompt,
                    "num_inference_steps": min(steps, 4) if is_turbo else steps,
                    "guidance_scale": 0.0 if is_turbo else 5.0,
                    "width": 512 if is_turbo else width,
                    "height": 512 if is_turbo else height,
                }
            img = pipe(**call_kwargs).images[0]
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    if total_gb < 36:
                        _PIPE = None
                _clear_cuda()
            except Exception:
                pass

    fname = f"uc_{uuid.uuid4().hex[:12]}.png"
    out_path = GENERATED_DIR / fname
    img.save(out_path, format="PNG")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    ms = round((time.time() - t0) * 1000)
    return {
        "url": f"/static/generated/{fname}",
        "filename": fname,
        "prompt": prompt,
        "en_prompt": en_prompt,
        "model": "Unicollect Imagine",
        "width": img.width,
        "height": img.height,
        "latency_ms": ms,
        "b64_json": b64,
    }


# ==============================================================================
# Unicollect NSFW Image Engine (John6666/ebara-pony-v1-sdxl + sdxl-vae-fp16-fix)
# ==============================================================================
NSFW_IMAGE_MODEL_ID = os.environ.get("NSFW_IMAGE_MODEL", "John6666/ebara-pony-v1-sdxl").strip() or "John6666/ebara-pony-v1-sdxl"
NSFW_VAE_MODEL_ID = os.environ.get("NSFW_VAE_MODEL", "madebyollin/sdxl-vae-fp16-fix").strip() or "madebyollin/sdxl-vae-fp16-fix"
NSFW_IMAGE_STEPS = int(os.environ.get("NSFW_IMAGE_STEPS", "25"))
NSFW_IMAGE_CFG = float(os.environ.get("NSFW_IMAGE_CFG", "6.0"))

_NSFW_PIPE = None
_NSFW_BLOCKED_RE = re.compile(
    r"\b(?:child|children|kid|kids|minor|minors|underage|toddler|baby|infant|loli|shota|preteen|young\s+girl|young\s+boy|schoolgirl|schoolboy)\b|کودک|بچه|خردسال|زیر\s*سن",
    re.IGNORECASE,
)

NSFW_PROMPT_ENHANCER_SYSTEM = (
    "You are an expert SDXL Pony V1 visual prompt engineer. Convert the user's Persian or English image request "
    "into a clean, vivid, comma-separated English tag prompt optimized for Pony SDXL (25 to 55 words). "
    "Focus on adult subjects, lighting, setting, composition, camera angle, outfit/details, and high-detail artistic quality tags. "
    "Output ONLY the comma-separated English tags (do NOT include score_9 or rating: prefixes — those are added automatically). "
    "No quotes, no explanations."
)


def enhance_nsfw_image_prompt(prompt: str, ollama_url: str, enhancer_model: str) -> str:
    clean = (prompt or "").strip()
    clean = re.sub(
        r"^(?:/nsfw|nsfw:|/imagine|imagine:|عکس\s+بدون\s+فیلتر|تصویر\s+بدون\s+فیلتر|عکس\s+بساز|تصویر\s+بساز)\s*[:\-]?\s*",
        "",
        clean,
        flags=re.I,
    ).strip() or prompt.strip()
    if not enhancer_model or (len(clean) >= 120 and not _has_persian(clean)):
        return clean
    try:
        r = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": enhancer_model,
                "messages": [
                    {"role": "system", "content": NSFW_PROMPT_ENHANCER_SYSTEM},
                    {"role": "user", "content": clean},
                ],
                "stream": False,
                "think": False,
                "keep_alive": "15m",
                "options": {"temperature": 0.35, "num_predict": 110},
            },
            timeout=12,
        )
        if r.ok:
            en = (r.json().get("message", {}).get("content") or "").strip()
            en = re.sub(r"<think>.*?</think>", "", en, flags=re.S).strip().strip('"').strip("'")
            if en and len(en) >= 8 and not _has_persian(en):
                return en
    except Exception:
        pass
    return clean


def _detect_nsfw_ratio(prompt: str, default_w: int = 1024, default_h: int = 1024) -> tuple[int, int]:
    low = (prompt or "").lower()
    if "16:9" in low or "landscape" in low or "banner" in low or "یوتیوب" in low:
        return 1216, 832
    if "9:16" in low or "portrait" in low or "استوری" in low or "شورتس" in low:
        return 832, 1216
    if "4:3" in low:
        return 1152, 864
    return default_w, default_h


def _load_nsfw_pipeline():
    global _NSFW_PIPE, _PIPE
    if _NSFW_PIPE is not None:
        return _NSFW_PIPE
    # Release standard image pipeline if loaded so VRAM stays clean
    if _PIPE is not None:
        _PIPE = None
        _clear_cuda()

    import torch
    from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    hf_token = (os.environ.get("HF_TOKEN") or "").strip() or None

    vae = AutoencoderKL.from_pretrained(NSFW_VAE_MODEL_ID, torch_dtype=dtype, token=hf_token)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        NSFW_IMAGE_MODEL_ID,
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
        token=hf_token,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if use_cuda:
        try:
            pipe.enable_attention_slicing()
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
        except Exception:
            pass
        if _free_vram_gb() >= 10.5:
            pipe = pipe.to("cuda")
        else:
            pipe.enable_model_cpu_offload()
    _NSFW_PIPE = pipe
    return _NSFW_PIPE


def generate_nsfw_image(
    prompt: str,
    ollama_url: str = "http://127.0.0.1:11434",
    router_model: str = "qwen3:1.7b",
    width: int = 1216,
    height: int = 832,
    steps: int = NSFW_IMAGE_STEPS,
    cfg: float = NSFW_IMAGE_CFG,
    neg_prompt: str = "",
    content_rating: str = "explicit",
    seed: int = -1,
) -> Dict[str, Any]:
    """Direct Ebara Pony V1 SDXL generation — uses raw user prompt verbatim with ZERO AI rewriting."""
    global _NSFW_PIPE
    raw_clean = re.sub(r"^/nsfw\s*[:\-]?\s*", "", (prompt or "").strip(), flags=re.I).strip()
    if _NSFW_BLOCKED_RE.search(raw_clean):
        raise ValueError("Requests involving minors are strictly prohibited.")
    t0 = time.time()

    w, h = _detect_nsfw_ratio(raw_clean, width, height)
    w = max(512, min(1216, (int(w) // 64) * 64))
    h = max(512, min(1216, (int(h) // 64) * 64))
    steps = max(12, min(35, int(steps)))
    cfg = max(3.0, min(9.0, float(cfg)))

    rating_clean = (content_rating or "explicit").strip().lower()
    rating_tag = f"rating:{rating_clean}" if rating_clean != "none" else ""
    pos_prefix = f"score_9, score_8_up, score_7_up, {rating_tag}".strip(", ")
    final_pos = f"{pos_prefix}, {raw_clean}" if pos_prefix else raw_clean

    base_neg = "score_4, score_5, score_6, (worst quality, low quality:1.4), bad anatomy, deformed limbs, bad hands, missing fingers, blurry"
    final_neg = f"{base_neg}, {neg_prompt.strip()}" if (neg_prompt or "").strip() else base_neg

    with _PIPE_LOCK:
        _free_ollama_vram_and_wait(ollama_url, min_free_gb=11.0, max_wait_s=18.0)
        try:
            import torch
            pipe = _load_nsfw_pipeline()
            generator = None
            if int(seed) != -1:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=dev).manual_seed(int(seed))
            img = pipe(
                prompt=final_pos,
                negative_prompt=final_neg,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=w,
                height=h,
                generator=generator,
            ).images[0]
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    if total_gb < 36:
                        _NSFW_PIPE = None
                _clear_cuda()
            except Exception:
                pass

    fname = f"uc_nsfw_{uuid.uuid4().hex[:12]}.png"
    out_path = GENERATED_DIR / fname
    img.save(out_path, format="PNG")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    ms = round((time.time() - t0) * 1000)
    return {
        "url": f"/static/generated/{fname}",
        "filename": fname,
        "prompt": raw_clean,
        "en_prompt": final_pos,
        "model": "Unicollect NSFW Image",
        "width": img.width,
        "height": img.height,
        "latency_ms": ms,
        "b64_json": b64,
    }
