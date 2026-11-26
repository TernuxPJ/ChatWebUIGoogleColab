<div align="center">
  <img src="docs/icon.png" alt="ChatWebUIGoogleColab Icon" width="96" height="96" />

  <h1>ChatWebUIGoogleColab</h1>
  <p><strong>Google Colab Edition of Unicollect ChatWebUI · 5-Way 4B Router, Pro 14B (All GPUs), Coder Ultra (Unlocked), Mini/1.2 &amp; SDXL-Turbo Image Generation over <code>trycloudflare</code></strong></p>

  [![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-OpenAI%20Router-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![Ollama](https://img.shields.io/badge/Ollama-5%20Models-111827?style=for-the-badge)](https://ollama.com/)
  [![Diffusers](https://img.shields.io/badge/SDXL--Turbo-Image%20Gen-8B5CF6?style=for-the-badge)](https://huggingface.co/stabilityai/sdxl-turbo)
  [![Google Colab](https://img.shields.io/badge/Google%20Colab-T4%20%7C%20L4%20%7C%20A100-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-3B82F6?style=for-the-badge)](./LICENSE)

  <br/>

  [Model Suite](#model-suite) • [Highlights](#highlights) • [Run in Google Colab](#run-in-google-colab) • [API Routes](#api-routes) • [License](#license)
</div>

---

<div align="center">

## Model Suite

</div>

| Panel Model | Backend Engine | GPU Compatibility | Role |
| :--- | :--- | :--- | :--- |
| **Unicollect Router 4B** | `qwen3:4b` | T4 / L4 / A100 | Fast 5-way hybrid router (`image`, `coder`, `science`, `math`, `chat`) & Persian→English image prompt translator |
| **Unicollect Chat / Math Pro 14B** | `qwen3:14b` | T4 / L4 / A100 | High-intelligence, fast all-GPU chat & mathematical reasoning |
| **Unicollect Chat / Mini** | `gpt-oss:20b` | T4 / L4 / A100 | Fast MoE general chat & bilingual EN/FA reasoning |
| **Unicollect 1.2 (Math & Science)** | `lfm2.5:24b` | L4 / A100 *(offload on T4)* | Deep step-by-step math & scientific derivations |
| **Unicollect Coder Ultra** | `qwen3-coder:30b` *(fallback `qwen3.6:35b-a3b`)* | L4 / A100 *(offload on T4)* | **Coding 100% unlocked** — full multi-file code, debugging & refactoring |
| **Unicollect Imagine** | `stabilityai/sdxl-turbo` | T4 / L4 / A100 | Real-time 4-step GPU image generation with inline chat display & OpenAI-compatible API |

---

<div align="center">

## Highlights

</div>

- **Smart 5-Way 4B Router (`qwen3:4b`):** Automatically detects and routes `image`, `coder`, `science`, `math`, and `chat` prompts in Persian and English.
- **High Context & Token Limits:** Configured for `32,768` context window (`OLLAMA_NUM_CTX`), `32,768` max output tokens for **Unicollect Coder Ultra**, `16,384` max output tokens for general chat, `64,000`-character input limit, and `32` conversation turns.
- **Coding Fully Unlocked for Coder Ultra:** Dedicated engineering system prompt with code-firewall bypass on SSE streaming and `/v1/chat/completions`, while preserving the no-code lock on standard chat models.
- **Built-in Image Generation (`app/image_gen.py`):** Generates images on GPU via `stabilityai/sdxl-turbo` (no Hugging Face token needed), translates Persian prompts automatically, manages VRAM safely on T4/L4/A100, and renders images inline inside chat bubbles.
- **Zero Redundant Downloads:** `colab/run_colab.py` checks Ollama storage on startup, skips models already on disk, and auto-downloads only missing models.

---

<div align="center">

## Run in Google Colab

</div>

1. Open [Google Colab](https://colab.research.google.com/) and choose **Runtime ➔ Change runtime type ➔ GPU (`T4`, `L4`, or `A100`)**.
2. Run a single cell:

```bash
!git clone https://github.com/TernuxPJ/ChatWebUIGoogleColab.git
%cd ChatWebUIGoogleColab
!pip install -q -r requirements.txt
!python3 -u colab/run_colab.py
```

The launcher installs `zstd`, `pciutils`, Ollama, `diffusers`, and `cloudflared` if missing, downloads only missing models, launches FastAPI on `127.0.0.1:8000`, and prints your live `https://*.trycloudflare.com` URL, `/v1` OpenAI endpoint, `/api/image` route, and Public API Key.

---

<div align="center">

## API Routes

</div>

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Persian RTL-first Web Chat UI with full model switcher (`Pro 14B`, `Mini`, `1.2`, `Coder Ultra`, `Imagine`) |
| `POST` | `/api/chat/stream` | SSE token-by-token streaming endpoint (supports text + inline image generation) |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions router |
| `GET` | `/v1/models` | List all available routed models |
| `POST` | `/api/image` | Direct JSON image generation endpoint |
| `POST` | `/v1/images/generations` | OpenAI-compatible image generation endpoint (`url` or `b64_json`) |
| `GET` | `/api/v1/key` · `/health` | Public API key bundle & live server/GPU status |

---

<div align="center">

## License

Released under the **[MIT License](./LICENSE)** — Copyright © 2026 **Muvixo**.

</div>
