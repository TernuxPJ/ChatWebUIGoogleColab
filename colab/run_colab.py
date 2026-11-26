#!/usr/bin/env python3
"""
ChatWebUIGoogleColab v15 Launcher
- Works on T4 (15 GB), L4 (24 GB), and A100 (40/80 GB) GPUs
- Step 1: Starts Ollama daemon
- Step 2: Downloads ALL missing Ollama models (skips already-installed ones)
- Step 3: Installs image generation dependencies
- Step 4: Pre-downloads image model weights + warms ALL Ollama models (pins Mini + Router in VRAM)
- Step 5: ONLY AFTER everything is downloaded & warmed, launches FastAPI + trycloudflare and prints the live URL
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

PORT = int(os.environ.get("PORT", "8000"))
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "qwen3:1.7b")
ENHANCER_MODEL = os.environ.get("ENHANCER_MODEL", ROUTER_MODEL)
PRO_MODEL = os.environ.get("PRO_MODEL", "qwen3:14b")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-oss:20b")
MATH_MODEL = os.environ.get("MATH_MODEL", "lfm2.5:24b")
CODER_MODEL = os.environ.get("CODER_MODEL", "qwen3-coder:30b")

PULL_MODELS = [
    m.strip()
    for m in os.environ.get(
        "PULL_MODELS",
        f"{ROUTER_MODEL},{PRO_MODEL},{MODEL_NAME},{MATH_MODEL},{CODER_MODEL}",
    ).split(",")
    if m.strip()
]


def run(cmd, check=True, env=None):
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}")
    return p.returncode


def detect_gpu_vram_gb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
        if out:
            name, mem_mb = [x.strip() for x in out[0].split(",", 1)]
            return name, float(mem_mb) / 1024.0
    except Exception:
        pass
    return "CPU", 0.0


def ollama_tags():
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as r:
            return [m.get("name", "") for m in json.loads(r.read().decode("utf-8")).get("models", [])]
    except Exception:
        return None


def ensure_ollama(vram_gb):
    if not shutil.which("zstd") or not shutil.which("lspci"):
        print("[1/6] Installing zstd & pciutils...", flush=True)
        run("apt-get update -qq && apt-get install -y -qq zstd pciutils")
    if not shutil.which("ollama"):
        print("[1/6] Installing Ollama...", flush=True)
        run("curl -fsSL https://ollama.com/install.sh | sh")

    installed = ollama_tags()
    if installed is not None:
        print(f"[2/6] Ollama already running. Stored models: {installed}", flush=True)
        return installed

    max_loaded = "2" if vram_gb >= 22 else "1"
    env = os.environ.copy()
    env.update({
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_NUM_GPU": os.environ.get("OLLAMA_NUM_GPU", "999"),
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KV_CACHE_TYPE": "q8_0",
        "OLLAMA_KEEP_ALIVE": "30m",
        "OLLAMA_MAX_LOADED_MODELS": os.environ.get("OLLAMA_MAX_LOADED_MODELS", max_loaded),
    })
    print(f"[2/6] Starting Ollama daemon (max_loaded={env['OLLAMA_MAX_LOADED_MODELS']})...", flush=True)
    subprocess.Popen(["ollama", "serve"], stdout=open("/tmp/ollama.log", "w"), stderr=subprocess.STDOUT, env=env)
    for _ in range(30):
        installed = ollama_tags()
        if installed is not None:
            print(f"[OK] Ollama ready. Stored models: {installed}", flush=True)
            return installed
        time.sleep(1)
    raise RuntimeError("Ollama failed to start. Check /tmp/ollama.log")


def pull_missing_models(installed):
    if os.environ.get("SKIP_PULL", "0").lower() in ("1", "true", "yes"):
        print("[3/6] SKIP_PULL=1 — using models already stored on disk.", flush=True)
        return
    inst_low = {m.strip().lower() for m in (installed or [])}
    missing = [
        m for m in PULL_MODELS
        if not any(x == m.lower() or x == f"{m.lower()}:latest" for x in inst_low)
    ]
    print(f"[3/6] Stored models: {sorted(inst_low) or 'none'}", flush=True)
    if not missing:
        print("[3/6] All required Ollama models are already downloaded.", flush=True)
        return
    print(f"[3/6] Downloading {len(missing)} missing model(s) before startup: {missing}", flush=True)
    for idx, m in enumerate(missing, 1):
        print(f"---> [{idx}/{len(missing)}] Pulling missing model: {m} ...", flush=True)
        rc = run(f"ollama pull {m}", check=False)
        if rc != 0 and m == "qwen3.6:35b-a3b":
            run("ollama pull qwen3-coder:30b", check=False)
        elif rc != 0:
            print(f"[WARN] Retrying pull for {m} once...", flush=True)
            run(f"ollama pull {m}", check=False)


def ensure_image_deps():
    try:
        import diffusers, accelerate, PIL, huggingface_hub  # noqa: F401
        print("[4/6] Image generation packages already installed.", flush=True)
    except ImportError:
        print("[4/6] Installing diffusers, transformers, accelerate, safetensors, pillow...", flush=True)
        run(f"{sys.executable} -m pip install -q -U diffusers transformers accelerate safetensors sentencepiece protobuf pillow huggingface_hub")


def ensure_cloudflared():
    existing = shutil.which("cloudflared")
    if existing:
        return existing
    print("[5/6] Downloading cloudflared...", flush=True)
    bin_path = "/usr/local/bin/cloudflared"
    run(f"curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o {bin_path}")
    os.chmod(bin_path, 0o755)
    return bin_path


def start_app_and_tunnel(cf_bin, image_model, warm_done):
    subprocess.run(["pkill", "-f", "uvicorn app.main:app"], check=False)
    subprocess.run(["pkill", "-x", "cloudflared"], check=False)
    time.sleep(1)

    print("[6/6] Starting ChatWebUIGoogleColab + trycloudflare quick tunnel...", flush=True)
    env = os.environ.copy()
    env.update({
        "ROUTER_MODEL": ROUTER_MODEL,
        "ENHANCER_MODEL": ENHANCER_MODEL,
        "PRO_MODEL": PRO_MODEL,
        "MODEL_NAME": MODEL_NAME,
        "MATH_MODEL": MATH_MODEL,
        "CODER_MODEL": CODER_MODEL,
        "IMAGE_MODEL": image_model,
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    if warm_done:
        # Prevent FastAPI background thread from re-warming / evicting hot models right after warm_all.py
        env["SKIP_STARTUP_PREWARM"] = "1"

    app_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=open("/tmp/chatwebui.log", "w"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    for _ in range(40):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    print(f"[OK] ChatWebUI listening on http://127.0.0.1:{PORT}", flush=True)
                    break
        except Exception:
            time.sleep(1)

    cf_proc = subprocess.Popen(
        [cf_bin, "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", f"http://127.0.0.1:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url_re = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    printed = False
    for line in cf_proc.stdout:
        m = url_re.search(line)
        if m and not printed:
            printed = True
            url = m.group(0)
            # Wait up to 20s for Cloudflare edge DNS to propagate so user never hits Error 1016
            print(f"[TUNNEL] Waiting for Cloudflare DNS propagation ({url})...", flush=True)
            for _ in range(20):
                try:
                    with urllib.request.urlopen(f"{url}/health", timeout=3) as hr:
                        if hr.status == 200:
                            break
                except Exception:
                    time.sleep(1)
            pub_key = open(".public_api_key").read().strip() if os.path.exists(".public_api_key") else "(see /api/v1/key)"
            print("\n" + "=" * 72, flush=True)
            print(f"  ChatWebUIGoogleColab LIVE URL : {url}", flush=True)
            print(f"  OpenAI-Compatible Base URL    : {url}/v1", flush=True)
            print(f"  Image Generation API          : {url}/api/image  &  {url}/v1/images/generations", flush=True)
            print(f"  Public API Key                : {pub_key}", flush=True)
            print(f"  Router/Enhancer: {ROUTER_MODEL} | Pro: {PRO_MODEL} | Image: {image_model}", flush=True)
            print(f"  Stored Ollama Models          : {ollama_tags()}", flush=True)
            print("=" * 72 + "\n", flush=True)
    app_proc.wait()


if __name__ == "__main__":
    g_name, g_vram = detect_gpu_vram_gb()
    print(f"[GPU] {g_name} ({g_vram:.1f} GB VRAM)", flush=True)

    default_img = "YuCollection/FLUX.1-schnell-Diffusers" if g_vram >= 36 else "SG161222/RealVisXL_V5.0_Lightning"
    image_model = os.environ.get("IMAGE_MODEL", default_img).strip() or default_img
    os.environ["IMAGE_MODEL"] = image_model
    os.environ.setdefault("WARM_IMAGE", "1")

    stored = ensure_ollama(g_vram)
    pull_missing_models(stored)
    ensure_image_deps()

    warm_done = False
    if os.environ.get("WARM_ALL", "1").lower() in ("1", "true", "yes"):
        warm_env = os.environ.copy()
        warm_env["IMAGE_MODEL"] = image_model
        warm_env.setdefault("WARM_IMAGE", "1")
        warm_env.setdefault("OLLAMA_MAX_LOADED_MODELS", "2" if g_vram >= 22 else "1")
        rc = run(f"{sys.executable} -u scripts/warm_all.py", check=False, env=warm_env)
        warm_done = (rc == 0)

    cf = ensure_cloudflared()
    start_app_and_tunnel(cf, image_model, warm_done)
