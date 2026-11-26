#!/usr/bin/env python3
"""
Unicollect Complete Warm-Up Script — ALL Models (v23)
Runs BEFORE FastAPI & trycloudflare print the live URL:
1. Waits for Ollama & pulls ANY missing Ollama models automatically
2. Downloads/caches ALL weights for the active ungated image model (RealVisXL_V5.0_Lightning or FLUX.1-schnell)
3. Unloads any leftover Ollama VRAM, runs a real GPU warm-up pass on Unicollect Imagine (caches CUDA/cuDNN kernels & verifies zero OOM), then releases 100% of PyTorch VRAM
4. Sequentially warms EVERY Ollama model (Coder Ultra, Chat 1.2, Chat Pro, plus any extra installed models) with its exact system prompt & waits for clean VRAM release between heavy models
5. Finishes by warming & PINNING Unicollect Chat Mini (exact KV-cache + think="low") and Unicollect Router (think=False) hot together in VRAM, then prints a complete status table
"""
import ast
import gc
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "qwen3:1.7b").strip()
ENHANCER_MODEL = os.environ.get("ENHANCER_MODEL", ROUTER_MODEL).strip()
PRO_MODEL = os.environ.get("PRO_MODEL", "qwen3:14b").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-oss:20b").strip()
MATH_MODEL = os.environ.get("MATH_MODEL", "lfm2.5:24b").strip()
CODER_MODEL = os.environ.get("CODER_MODEL", "qwen3-coder:30b").strip()
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "SG161222/RealVisXL_V5.0_Lightning").strip()
NSFW_IMAGE_MODEL = os.environ.get("NSFW_IMAGE_MODEL", "John6666/ebara-pony-v1-sdxl").strip()
NSFW_VAE_MODEL = os.environ.get("NSFW_VAE_MODEL", "madebyollin/sdxl-vae-fp16-fix").strip()
WARM_NSFW_IMAGE = os.environ.get("WARM_NSFW_IMAGE", "1").lower() in ("1", "true", "yes")

WARM_IMAGE = os.environ.get("WARM_IMAGE", "1").lower() in ("1", "true", "yes")
WARM_IMAGE_GPU = os.environ.get("WARM_IMAGE_GPU", "1").lower() in ("1", "true", "yes")
AUTO_PULL_MISSING = os.environ.get("AUTO_PULL_MISSING", "1").lower() in ("1", "true", "yes")
NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
CODER_NUM_CTX = int(os.environ.get("CODER_NUM_CTX", "32768"))
MAX_REPLY_TOKENS = int(os.environ.get("MAX_REPLY_TOKENS", "8192"))
CODER_REPLY_TOKENS = int(os.environ.get("CODER_REPLY_TOKENS", "32768"))
MAX_LOADED = int(os.environ.get("OLLAMA_MAX_LOADED_MODELS", "2") or "2")


def _extract_prompt_var(var_name: str, label: str) -> str:
    if var_name == "DEFAULT_SYSTEM_PROMPT":
        env_sp = (os.environ.get("SYSTEM_PROMPT") or "").strip()
        if env_sp:
            return env_sp.replace("{MODEL_NAME}", label)
    try:
        main_py = (Path(__file__).resolve().parent.parent / "app" / "main.py").read_text(encoding="utf-8")
        tree = ast.parse(main_py)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        val = ast.literal_eval(node.value)
                        if isinstance(val, str):
                            return val.replace("{MODEL_NAME}", label)
    except Exception:
        pass
    return f"You are {label}, created by Muvixo."


def wait_for_ollama(timeout_s: int = 40) -> list[str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
                return [m.get("name", "") for m in json.loads(r.read().decode("utf-8")).get("models", [])]
        except Exception:
            time.sleep(1)
    return []


def get_installed_ollama() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            return [m.get("name", "") for m in json.loads(r.read().decode("utf-8")).get("models", [])]
    except Exception:
        return []


def get_loaded_ollama() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/ps", timeout=5) as r:
            return [m.get("name", "") for m in json.loads(r.read().decode("utf-8")).get("models", [])]
    except Exception:
        return []


def is_installed(model: str, installed_low: set[str]) -> bool:
    if not model:
        return False
    ml = model.strip().lower()
    return any(x == ml or x == f"{ml}:latest" for x in installed_low)


def unload_all_ollama_and_wait(timeout_s: float = 15.0):
    loaded = get_loaded_ollama()
    for name in loaded:
        for ep, body in (
            ("/api/generate", {"model": name, "keep_alive": 0}),
            ("/api/chat", {"model": name, "messages": [], "keep_alive": 0}),
        ):
            try:
                req = urllib.request.Request(
                    f"{OLLAMA_URL}{ep}",
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=10) as r:
                    _ = r.read()
            except Exception:
                pass
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not get_loaded_ollama():
            break
        time.sleep(0.5)


def unload_model_and_wait(model: str, timeout_s: float = 12.0):
    if not model:
        return
    for ep, body in (
        ("/api/generate", {"model": model, "keep_alive": 0}),
        ("/api/chat", {"model": model, "messages": [], "keep_alive": 0}),
    ):
        try:
            req = urllib.request.Request(
                f"{OLLAMA_URL}{ep}",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                _ = r.read()
        except Exception:
            pass
    ml = model.strip().lower()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        loaded_low = {x.strip().lower() for x in get_loaded_ollama()}
        if ml not in loaded_low and f"{ml}:latest" not in loaded_low:
            break
        time.sleep(0.5)


def ensure_all_ollama_present(required_models: list[str]) -> set[str]:
    installed = wait_for_ollama(40)
    inst_low = {m.strip().lower() for m in installed}
    if not AUTO_PULL_MISSING:
        return inst_low
    seen = []
    for m in required_models:
        if m and m not in seen:
            seen.append(m)
    for m in seen:
        if not is_installed(m, inst_low):
            print(f"[PULL] Missing Ollama model: {m} — downloading now...", flush=True)
            rc = subprocess.call(["ollama", "pull", m])
            if rc != 0:
                print(f"[WARN] Retrying pull for {m}...", flush=True)
                subprocess.call(["ollama", "pull", m])
    return {m.strip().lower() for m in get_installed_ollama()}


def warm_image_model() -> tuple[bool, str]:
    if not WARM_IMAGE or not IMAGE_MODEL:
        print("[SKIP] Image model warm-up disabled (WARM_IMAGE=0).", flush=True)
        return False, "skipped"

    t0 = time.time()
    hf_token = (os.environ.get("HF_TOKEN") or "").strip() or None
    candidates = []
    for c in (IMAGE_MODEL, "SG161222/RealVisXL_V5.0_Lightning", "stabilityai/sdxl-turbo"):
        if c and c not in candidates and "black-forest-labs/" not in c.lower():
            candidates.append(c)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[WARN] huggingface_hub not installed; skipping image model warm-up.", flush=True)
        return False, "missing huggingface_hub"

    cached_repo = None
    for repo_id in candidates:
        print(f"[WARM] Unicollect Imagine ({repo_id}) — downloading/verifying weights ...", flush=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                token=hf_token,
                ignore_patterns=["*.ckpt", "*.onnx", "*.onnx_data", "*openvino*", "*flax*", "*.msgpack"],
            )
            cached_repo = repo_id
            print(f"[OK]   Image weights cached ({repo_id}) in {time.time() - t0:.1f}s", flush=True)
            break
        except Exception as exc:
            print(f"[WARN] Could not cache {repo_id}: {exc}", flush=True)

    if not cached_repo:
        return False, "download failed"

    if not WARM_IMAGE_GPU:
        return True, f"{cached_repo} (disk cached)"

    # Run a fast 2-step GPU warm-up pass so CUDA/cuDNN kernels are ready & verified, then release 100% VRAM
    unload_all_ollama_and_wait(15.0)
    t_gpu = time.time()
    pipe = None
    try:
        import torch
        if not torch.cuda.is_available():
            return True, f"{cached_repo} (CPU cached)"
        print(f"[WARM] Unicollect Imagine ({cached_repo}) — running 2-step GPU kernel warm-up ...", flush=True)
        low = cached_repo.lower()
        if "flux" in low:
            from diffusers import FluxPipeline
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            pipe = FluxPipeline.from_pretrained(cached_repo, torch_dtype=dtype, token=hf_token)
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_gb >= 36:
                pipe = pipe.to("cuda")
            else:
                pipe.enable_sequential_cpu_offload()
            _ = pipe(prompt="red apple studio photo", num_inference_steps=2, guidance_scale=0.0, width=512, height=512, max_sequence_length=64).images[0]
        else:
            from diffusers import DiffusionPipeline
            kw = {"torch_dtype": torch.float16, "token": hf_token}
            if "realvis" in low or "turbo" in low:
                kw["variant"] = "fp16"
            try:
                pipe = DiffusionPipeline.from_pretrained(cached_repo, **kw)
            except Exception:
                kw.pop("variant", None)
                pipe = DiffusionPipeline.from_pretrained(cached_repo, **kw)
            try:
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
            except Exception:
                pass
            free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
            if free_gb >= 10.5:
                pipe = pipe.to("cuda")
            else:
                pipe.enable_model_cpu_offload()
            _ = pipe(prompt="red apple studio photo", num_inference_steps=2, guidance_scale=1.5, width=512, height=512).images[0]
        print(f"[OK]   Unicollect Imagine GPU warm-up verified in {time.time() - t_gpu:.1f}s", flush=True)
        return True, f"{cached_repo} (GPU verified)"
    except Exception as exc:
        print(f"[WARN] Image GPU warm-up skipped/failed (weights still cached): {exc}", flush=True)
        return True, f"{cached_repo} (cached)"
    finally:
        if pipe is not None:
            del pipe
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass



def warm_nsfw_image_model() -> tuple[bool, str]:
    if not WARM_IMAGE or not WARM_NSFW_IMAGE or not NSFW_IMAGE_MODEL:
        print("[SKIP] Unicollect NSFW Image pre-download disabled.", flush=True)
        return False, "skipped"
    t0 = time.time()
    hf_token = (os.environ.get("HF_TOKEN") or "").strip() or None
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False, "missing huggingface_hub"
    try:
        print(f"[WARM] Unicollect NSFW VAE ({NSFW_VAE_MODEL}) — downloading/verifying ...", flush=True)
        snapshot_download(repo_id=NSFW_VAE_MODEL, token=hf_token, ignore_patterns=["*.ckpt", "*.onnx", "*.msgpack"])
        print(f"[WARM] Unicollect NSFW Image ({NSFW_IMAGE_MODEL}) — downloading/verifying weights ...", flush=True)
        snapshot_download(
            repo_id=NSFW_IMAGE_MODEL,
            token=hf_token,
            ignore_patterns=["*.ckpt", "*.onnx", "*.onnx_data", "*openvino*", "*flax*", "*.msgpack"],
        )
        print(f"[OK]   Unicollect NSFW Image cached ({NSFW_IMAGE_MODEL}) in {time.time() - t0:.1f}s", flush=True)
        return True, f"{NSFW_IMAGE_MODEL} (cached)"
    except Exception as exc:
        print(f"[WARN] Could not cache NSFW image model {NSFW_IMAGE_MODEL}: {exc}", flush=True)
        return False, "failed"


def warm_ollama_model(
    label: str,
    model: str,
    installed_low: set[str],
    *,
    prompt_var: str = "DEFAULT_SYSTEM_PROMPT",
    pin: bool = False,
    is_router: bool = False,
    is_coder: bool = False,
) -> tuple[bool, float]:
    if not model:
        return False, 0.0
    if not is_installed(model, installed_low):
        print(f"[SKIP] {label} ({model}) is not installed.", flush=True)
        return False, 0.0

    t0 = time.time()
    mode_tag = "PIN IN VRAM" if pin else "WARM + PRE-CACHE"
    print(f"[WARM] {label} ({model}) [{mode_tag}] ...", flush=True)

    ml = model.lower()
    is_gpt_oss = ml.startswith("gpt-oss")
    is_lfm = ml.startswith("lfm")

    if is_router:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "think": False,
            "keep_alive": -1 if pin else "10s",
            "options": {"num_ctx": 2048, "num_predict": 4, "temperature": 0.0},
        }
    else:
        sys_prompt = _extract_prompt_var(prompt_var, label)
        think_val = "low" if is_gpt_oss else (False if is_coder else True)
        # Match chat_stream's EXACT options dict so KV cache prefix is reused 100%
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "hi"},
            ],
            "stream": True,
            "think": think_val,
            "keep_alive": -1 if pin else "10s",
            "options": {
                "temperature": 0.35 if is_lfm else 0.55,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.08,
                "num_ctx": CODER_NUM_CTX if is_coder else NUM_CTX,
                "num_predict": CODER_REPLY_TOKENS if is_coder else (MAX_REPLY_TOKENS if MAX_REPLY_TOKENS > 0 else 8192),
            },
        }

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            if payload.get("stream"):
                # Wait until the model actually emits its first real token (proves weights + KV cache are built on GPU)
                for raw in r:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue
                    msg = obj.get("message") or {}
                    if msg.get("content") or msg.get("thinking") or obj.get("done"):
                        break
            else:
                _ = r.read()
        elapsed = time.time() - t0
        if not pin:
            unload_model_and_wait(model, 12.0)
        print(f"[OK]   {label} ({model}) verified in {elapsed:.1f}s ({'pinned in VRAM' if pin else 'cached & unloaded'})", flush=True)
        return True, elapsed
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"[ERR]  {label} ({model}) warm-up failed ({elapsed:.1f}s): {exc}", flush=True)
        unload_model_and_wait(model, 8.0)
        return False, elapsed


def main():
    print("=" * 74, flush=True)
    print("  Unicollect Complete Warm-Up — ALL Models (Ollama + Image GPU)", flush=True)
    print("=" * 74, flush=True)

    # 1. Ensure ALL required Ollama models are downloaded first
    required_ollama = [ROUTER_MODEL, ENHANCER_MODEL, PRO_MODEL, MODEL_NAME, MATH_MODEL, CODER_MODEL]
    installed_low = ensure_all_ollama_present(required_ollama)
    print(f"[INFO] Installed Ollama models: {sorted(installed_low) or 'none'}", flush=True)

    # 2. Download + GPU-warm Unicollect Imagine (while Ollama VRAM is empty), then release PyTorch VRAM
    img_ok, img_info = warm_image_model()
    nsfw_ok, nsfw_info = warm_nsfw_image_model()
    summary = [
        ("Unicollect Imagine", img_info, "OK" if img_ok else "WARN"),
        ("Unicollect NSFW Image", nsfw_info, "OK" if nsfw_ok else "WARN"),
    ]

    # 3. Warm ALL secondary Ollama models (exact prompts, clean unload-wait between each)
    warmed_set = set()
    secondary = [
        ("Unicollect Coder Ultra", CODER_MODEL, "CODER_SYSTEM_PROMPT", True),
        ("Unicollect Chat 1.2", MATH_MODEL, "MATH_SYSTEM_PROMPT", False),
        ("Unicollect Chat Pro", PRO_MODEL, "DEFAULT_SYSTEM_PROMPT", False),
    ]
    if ENHANCER_MODEL and ENHANCER_MODEL not in (ROUTER_MODEL, MODEL_NAME, PRO_MODEL, MATH_MODEL, CODER_MODEL):
        secondary.append(("Unicollect Enhancer", ENHANCER_MODEL, "DEFAULT_SYSTEM_PROMPT", False))

    for label, model, pvar, is_coder in secondary:
        if model and model.lower() not in warmed_set and model not in (MODEL_NAME, ROUTER_MODEL):
            ok, sec = warm_ollama_model(label, model, installed_low, prompt_var=pvar, pin=False, is_coder=is_coder)
            summary.append((label, model, f"OK ({sec:.1f}s cached)" if ok else "FAILED"))
            warmed_set.add(model.lower())

    # Warm any extra installed Ollama models not in the standard list
    pinned_low = {m.lower() for m in (MODEL_NAME, ROUTER_MODEL) if m}
    for extra in sorted(installed_low):
        base = extra.removesuffix(":latest")
        if extra not in warmed_set and base not in warmed_set and extra not in pinned_low and base not in pinned_low:
            ok, sec = warm_ollama_model(f"Extra ({extra})", extra, installed_low, prompt_var="DEFAULT_SYSTEM_PROMPT", pin=False)
            summary.append(("Extra Ollama", extra, f"OK ({sec:.1f}s cached)" if ok else "FAILED"))
            warmed_set.add(extra)

    # 4. Warm & PIN everyday hot models in VRAM (Mini + Router on L4/A100, Mini on T4)
    ok_m, sec_m = warm_ollama_model("Unicollect Chat Mini", MODEL_NAME, installed_low, prompt_var="DEFAULT_SYSTEM_PROMPT", pin=True)
    summary.append(("Unicollect Chat Mini", MODEL_NAME, f"PINNED ({sec_m:.1f}s)" if ok_m else "FAILED"))

    if MAX_LOADED >= 2 and ROUTER_MODEL and ROUTER_MODEL != MODEL_NAME:
        ok_r, sec_r = warm_ollama_model("Unicollect Router", ROUTER_MODEL, installed_low, pin=True, is_router=True)
        summary.append(("Unicollect Router", ROUTER_MODEL, f"PINNED ({sec_r:.1f}s)" if ok_r else "FAILED"))

    print("-" * 74, flush=True)
    for lbl, mdl, st in summary:
        print(f"  {lbl:<22} | {mdl:<36} | {st}", flush=True)
    print(f"[HOT]  Pinned in VRAM right now: {get_loaded_ollama()}", flush=True)
    print("=" * 74, flush=True)


if __name__ == "__main__":
    main()
