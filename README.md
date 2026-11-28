<div align="center">
  <img src="docs/icon.png" alt="ChatWebUIGoogleColab Icon" width="96" height="96" />

  <h1>ChatWebUIGoogleColab</h1>
  <p><strong>Google Colab Edition of Unicollect ChatWebUI · Multi-Mode AI Chat, Smart Routing, Coding Assistant &amp; Real-Time Image Generation</strong></p>

  [![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-OpenAI%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU%20Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-3B82F6?style=for-the-badge)](./LICENSE)

  <br/>

  [Unicollect Suite](#unicollect-suite) • [Highlights](#highlights) • [Run in Google Colab](#run-in-google-colab) • [API Routes](#api-routes) • [License](#license)
</div>

---

<div align="center">

## Unicollect Suite

</div>

| Mode | GPU Support | Best For |
| :--- | :--- | :--- |
| **Unicollect Auto** | T4 / L4 / A100 | Automatic smart routing across chat, math, science, coding, and image generation |
| **Unicollect Pro** | T4 / L4 / A100 | High-accuracy everyday conversation, writing, and mathematical reasoning |
| **Unicollect Mini** | T4 / L4 / A100 | Ultra-fast responses and bilingual Persian / English chat |
| **Unicollect 1.2** | L4 / A100 *(T4 supported)* | Deep step-by-step math, physics, and scientific analysis |
| **Unicollect Coder Ultra** | L4 / A100 *(T4 supported)* | Full multi-file software engineering, debugging, and refactoring |
| **Unicollect Imagine** | T4 / L4 / A100 | Fast GPU image generation with automatic Persian-to-English prompt translation |

---

<div align="center">

## Highlights

</div>

- **Smart Automatic Routing:** Detects whether your prompt is general chat, math, science, software engineering, or image generation in Persian and English and dispatches it automatically.
- **Long Context & High Output Limits:** Supports extended multi-turn conversations, long prompts, and large code outputs smoothly over SSE token streaming.
- **Built-In Image Generation:** Generate images directly inside chat bubbles or through the REST API, with automatic Persian prompt translation and safe GPU memory management.
- **OpenAI-Compatible API:** Ready-to-use `/v1/chat/completions`, `/v1/models`, and `/v1/images/generations` endpoints with an instant public HTTPS tunnel.
- **Fast Startup:** Automatically checks local storage on launch and skips already-installed components.

---

<div align="center">

## Run in Google Colab

</div>

1. Open [Google Colab](https://colab.research.google.com/) and select **Runtime ➔ Change runtime type ➔ GPU (`T4`, `L4`, or `A100`)**.
2. Run a single cell:

```bash
!git clone https://github.com/TernuxPJ/ChatWebUIGoogleColab.git
%cd ChatWebUIGoogleColab
!pip install -q -r requirements.txt
!python3 -u colab/run_colab.py
```

The launcher sets up the runtime automatically, starts the web panel and API server on `127.0.0.1:8000`, and prints your live public URL, `/v1` API endpoint, `/api/image` route, and Public API Key.

---

<div align="center">

## API Routes

</div>

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Bilingual RTL/LTR Web Chat UI with full Unicollect mode switcher |
| `POST` | `/api/chat/stream` | Real-time SSE token-by-token streaming (text + inline image generation) |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions endpoint |
| `GET` | `/v1/models` | List all available Unicollect modes |
| `POST` | `/api/image` | Direct JSON image generation endpoint |
| `POST` | `/v1/images/generations` | OpenAI-compatible image generation endpoint (`url` or `b64_json`) |
| `GET` | `/api/v1/key` · `/health` | Public API key bundle &amp; live server status |

---

<div align="center">

## License

Released under the **[MIT License](./LICENSE)** — Copyright © 2026 **Muvixo**.

</div>
