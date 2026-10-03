# ⚡ WarpGate

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Go Report Card](https://goreportcard.com/badge/github.com/Neuxbane/Ollama-One)](https://goreportcard.com/report/github.com/Neuxbane/Ollama-One)
[![Build State](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

> **A strictly stateless, high-performance universal AI gateway** that exposes **Google Gemini** and **DeepSeek** through the **OpenAI Chat Completions API** — with native tool calling, local semantic tool retrieval (`:warp_tools`), vision, extended thinking, streaming, and dynamic model middleware.

Point any OpenAI-compatible client (VS Code Copilot, the OpenAI SDK, LangChain, etc.) at WarpGate and talk to Gemini or DeepSeek without changing your integration. No API keys are stored on the server — every request carries its own.

---

## Table of Contents

- [Key Capabilities](#-key-capabilities)
- [Architecture](#️-architecture)
- [Model Routing](#-model-routing)
- [Quick Start](#-quick-start)
- [Authentication](#-authentication-stateless)
- [API Reference](#-api-reference)
- [Dynamic Model Middleware](#-dynamic-model-middleware)
- [Connect VS Code Copilot](#️-connect-vs-code-copilot)
- [Extending WarpGate](#-extending-warpgate)
- [Project Layout](#-project-layout)
- [Contributing](#-contributing)

---

## 🎯 Key Capabilities

| | Capability |
|---|---|
| 🛡️ | **100% Stateless by Default** — No `config.json` required. Supply your provider API key per request. |
| ⚡ | **WarpTools Semantic Search** — `:warp_tools(top_k=10)` dynamically searches 100+ tools locally via vector embeddings. |
| 🧠 | **Extended Thinking** — Map OpenAI `reasoning_effort` (`low` / `medium` / `high`) to Gemini thinking budgets and DeepSeek-R1 reasoning deltas. |
| 🛠️ | **Agent Tool Calling** — Full OpenAI tool-call compatibility, including streaming deltas and `finish_reason: tool_calls`. |
| 👁️ | **Multimodal Vision** — Analyze images and screenshots via `image_url` objects or `data:` URIs. |
| ⚡ | **Streaming (SSE)** — Server-Sent Events with correct role priming, usage reporting, and `[DONE]` termination. |
| 🧩 | **Dynamic Model Middleware** — Embed directives like `:warp_tools(top_k=10)` and `:truncate(all, 4000)` directly in the model ID. |
| 📦 | **Embedded UI & Assets** — The landing page and docs are compiled into the binary; no external files needed at runtime. |
| 🐳 | **One-Command Build** — `./build.sh` builds the binary and exports a portable compressed Docker image. |

---

## 🏗️ Architecture

```
 OpenAI-compatible client
        │
        │  POST /v1/chat/completions   (Authorization: Bearer <key>)
        ▼
 ┌───────────────────────────────────────────────┐
 │                   WarpGate                    │
 │  • Auth extraction (header / query / body)     │
 │  • OpenAI ⇄ Gemini / DeepSeek translation     │
 │  • WarpTools (:warp_tools local semantic RAG) │
 │  • Model middleware (:truncate, …)             │
 │  • Streaming (SSE) & tool-call handling        │
 └───────────────────────────────────────────────┘
        │               │                  │
        │ google/...    │ google/ws/...    │ deepseek/...
        ▼               ▼                  ▼
  Gemini REST API   Gemini Live (WS)  DeepSeek REST API
```

The proxy is **stateless**: it keeps no API keys, and sessions are only held in memory when a client explicitly opts in via `session_id`.

---

## 🌐 Model Routing

Routing is determined strictly by the model ID prefix or model name:

| Prefix / Pattern | Transport | Example |
|---|---|---|
| `deepseek/<model>` or `deepseek-*` | DeepSeek REST API (OpenAI Format) | `deepseek/deepseek-chat`, `deepseek-reasoner` |
| `google/ws/<model>` | Gemini Live WebSocket (Bidi) | `google/ws/gemini-3.1-flash-live-preview` |
| `google/<model>` or `<model>` | Gemini REST API | `google/gemini-2.0-flash` |

### DeepSeek Models (`deepseek/…` or `deepseek-*`)

Base URL: `https://api.deepseek.com` (configurable via `DEEPSEEK_BASE_URL`)

- `deepseek-chat` or `deepseek/deepseek-chat` (DeepSeek-V3 — chat, completions, tool calling, JSON mode)
- `deepseek-reasoner` or `deepseek/deepseek-reasoner` (DeepSeek-R1 — reasoning and extended thinking via `reasoning_content`)

Supply your DeepSeek API key per request via `Authorization: Bearer <key>`, `x-api-key`, query parameter, or set `DEEPSEEK_API_KEY`.

### WebSocket Live models (`google/ws/…`)

- `google/ws/gemini-3.8-live-extended-thinking`
- `google/ws/gemini-3.8-live`
- `google/ws/gemini-3.1-flash-live-preview`
- `google/ws/gemini-2.5-flash-native-audio-preview-09-2025`
- `google/ws/gemini-2.5-flash-native-audio-preview-12-2025`

### REST models (`google/…` or bare)

Any valid Gemini REST model works, for example:

- `google/gemini-2.5-pro`
- `google/gemini-2.5-flash`
- `google/gemini-2.0-flash`

> Call `GET /v1/models` to retrieve the list of available models.

---

## 🚀 Quick Start

### Option 1 — Docker Compose (stateless, recommended)

```bash
docker compose up -d --build
```

The proxy starts on `http://127.0.0.1:11434`.

### Option 2 — Build & export a portable image

```bash
./build.sh
```

This produces:

- **`warpgate`** — native standalone executable
- **`warpgate-docker.tar.gz`** — compressed, portable Docker image

Load and run it anywhere:

```bash
docker load < warpgate-docker.tar.gz
docker run -d -p 11434:11434 --name warpgate warpgate:latest
```

> 💡 **Tip:** skip the Docker step with `./build.sh --no-docker`, or build and immediately run with `./build.sh --run`.

### Option 3 — Optional server-wide config

To provide a fallback key on the server instead of per request, create `config.json` (see [`config.example.json`](config.example.json)):

```json
[
  {
    "type": "google",
    "key": "AIzaSy_YOUR_GOOGLE_GEMINI_API_KEY_HERE"
  }
]
```

You can also set the `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `DEEPSEEK_API_KEY` environment variable.
Optionally override the DeepSeek base URL with `DEEPSEEK_BASE_URL` (defaults to `https://api.deepseek.com`).

### Option 4 — Run locally

```bash
go build -o warpgate .
./warpgate
```

Set `PORT` (or `OLLAMA_PORT`) to override the default port `11434`, and `HOST` to override the default bind address `0.0.0.0`.

---

## 🔑 Authentication (Stateless)

Because the proxy stores no keys, clients pass the Gemini or DeepSeek API key with each request. The following locations are checked, in order:

1. **Standard `Authorization` header**
   ```http
   Authorization: Bearer AIzaSy...
   ```
2. **`x-api-key` header** (Anthropic / Ollama style)
   ```http
   x-api-key: AIzaSy...
   ```
3. **`X-Goog-Api-Key` header** (Google Cloud style)
   ```http
   X-Goog-Api-Key: AIzaSy...
   ```
4. **`api-key` header** (Azure / generic)
   ```http
   api-key: AIzaSy...
   ```
5. **URL query parameter**
   ```http
   POST /v1/chat/completions?key=AIzaSy...
   POST /v1/chat/completions?api_key=AIzaSy...
   ```
6. **JSON body**
   ```json
   {
     "model": "gemini-2.0-flash",
     "api_key": "AIzaSy...",
     "messages": [...]
   }
   ```

> Placeholder values such as `dummy`, `none`, `null`, `ollama`, `test`, and `sk-placeholder` are ignored, so clients that require a non-empty key still work.

If no usable key is found in the request, the proxy falls back to a configured `config.json` / environment key.

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Create a chat completion (streaming or non-streaming). |
| `GET` | `/v1/models` | List available Gemini models. |
| `GET` | `/v1/models/{id}` | Retrieve a single model descriptor. |
| `GET` | `/healthz` | Liveness probe — returns `{"status":"ok"}`. |
| `GET` | `/` | Web UI with documentation (embedded in the binary). |

Aliases without the `/v1` prefix (`/chat/completions`, `/models`) are also served for maximum client compatibility. Unknown routes return a structured JSON `404`.

### Example request

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Authorization: Bearer $GEMINI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.0-flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Example: Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_GEMINI_API_KEY",
    base_url="http://127.0.0.1:11434/v1",
)

resp = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

### Extended Thinking

Set `reasoning_effort` to `low`, `medium`, or `high` to enable Gemini thinking output:

```json
{
  "model": "gemini-2.5-pro",
  "reasoning_effort": "high",
  "messages": [{"role": "user", "content": "Solve this step by step…"}]
}
```

---

## 🧩 Dynamic Model Middleware

Append `:function(args)` directives to a model ID to transform the request on the fly. The base model is resolved automatically and each directive is executed in order.

```
REAL_MODEL_ID:FUNCTION_NAME(ARG1, ARG2, …)
```

Example — Gemini 2.0 Flash with every part truncated to 4000 characters:

```
gemini-2.0-flash:truncate(all, 4000)
```

### Built-in directive — `:truncate(target, limit)`

Enforce maximum character limits on prompt input and tool/function responses to prevent token overflow.

| Directive | Scope | Description |
|---|---|---|
| `:truncate(all, 4000)` | all | Truncate system prompt, user, assistant, and tool/function content. |
| `:truncate(tools, 2000)` | tools | Truncate only tool/function responses and arguments. |
| `:truncate(user, 1000)` | user | Truncate only user messages. |
| `:truncate(assistant, 1000)` | assistant | Truncate only assistant messages. |
| `:truncate(system, 1000)` | system | Truncate system messages / instructions. |
| `:truncate(4000)` | all | Shorthand for `:truncate(all, 4000)`. |

- **Defaults:** `target = all`, `limit = 4000`.
- **Safety:** truncation is unicode/rune-safe and appends a marker (e.g. `... [truncated N chars]`), guaranteeing the result never exceeds the limit.

### Built-in directive — `:warp_tools(top_k=10)`

Prevent prompt bloat and token exhaustion when clients send 100+ tools. Instead of feeding massive schemas upfront:
- If the incoming tool count is **$\le$ `top_k`**, the tools pass through directly to the model without overhead.
- If the incoming tool count **$>$ `top_k`**, the proxy swaps all tools for two lightweight meta-tools:
  - `searchTools(query)`: The model searches for capabilities using semantic queries. The proxy runs a **fast local semantic search vector index** (cosine similarity + TF-IDF subword matching) and injects only the top matching tools.
  - `execute(tools=[{name, arguments}])`: The model invokes the chosen tool(s), and the proxy transparently unwraps them into native OpenAI tool calls for the client.

| Directive | Description |
|---|---|
| `:warp_tools(top_k=10)` | Dynamic tool retrieval with custom threshold `top_k=10`. |
| `:warp_tools(5)` | Shorthand for `:warp_tools(top_k=5)`. |
| `:warp_tools()` | Default `top_k=10`. |

Example:
```json
{
  "model": "deepseek-chat:warp_tools(top_k=10)",
  "messages": [{"role": "user", "content": "Book flight and check weather in Tokyo"}],
  "tools": [/* 100+ tools */]
}
```

---

## 🛠️ Connect VS Code Copilot

### Step 1 — Open Model Management

In the Copilot Chat panel, click the model dropdown, then choose **Manage Models…**.

![Step 1 - Manage Models](assets/step1.png)

### Step 2 — Open the Language Models JSON

In the top-right of the Language Models settings tab, click the **Open Language Models (JSON)** icon.

![Step 2 - Open Language Models JSON](assets/step2.png)

### Step 3 — Add the custom endpoint

Paste the configuration below into `chatLanguageModels.json`.

![Step 3 - Configuration Overview](assets/step3.png)

```json
{
  "name": "WarpGate",
  "vendor": "customendpoint",
  "apiType": "chat-completions",
  "models": [
    {
      "id": "deepseek-chat:warp_tools(top_k=10)",
      "name": "DeepSeek V3 (WarpTools)",
      "url": "https://proxy.ai.sesh.top/v1/chat/completions",
      "toolCalling": true,
      "maxInputTokens": 64000,
      "maxOutputTokens": 8000
    },
    {
      "id": "google/ws/gemini-3.1-flash-live-preview:truncate(all, 4000)",
      "name": "Gemini 3.1 Flash",
      "url": "https://proxy.ai.sesh.top/v1/chat/completions",
      "toolCalling": true,
      "vision": true,
      "supportsReasoningEffort": ["none", "low", "medium", "high"],
      "reasoningEffortFormat": "chat-completions",
      "maxInputTokens": 65000,
      "maxOutputTokens": 65000
    },
    {
      "id": "google/ws/gemini-3.8-live-extended-thinking:truncate(tools, 4000)",
      "name": "Gemini 3.8 Flash",
      "url": "https://proxy.ai.sesh.top/v1/chat/completions",
      "toolCalling": true,
      "vision": true,
      "supportsReasoningEffort": ["none", "low", "medium", "high"],
      "reasoningEffortFormat": "chat-completions",
      "maxInputTokens": 128000,
      "maxOutputTokens": 16000
    }
  ]
}
```

### Extended Thinking controls

To expose `low` / `medium` / `high` reasoning effort in the chat input bar, ensure each thinking-capable model includes:

```json
"supportsReasoningEffort": ["low", "medium", "high"],
"reasoningEffortFormat": "chat-completions"
```

![Extended Thinking Configuration](assets/extended_thinking_config.png)

### Agent Mode in action

![VS Code Copilot Agent in Action](assets/chatScreenshoot.png)

---

## 🧱 Extending WarpGate

Middleware lives in the [`middleware/`](middleware/) package. To add a new directive:

1. Define a `ModifierFunc`:
   ```go
   func MyModifier(req *providers.CompletionRequest, args []string) error {
       // mutate req in place
       return nil
   }
   ```
2. Register it in an `init()`:
   ```go
   func init() {
       Register("mydirective", MyModifier)
   }
   ```

It becomes available immediately as `model:mydirective(arg1, arg2)`.

---

## 📁 Project Layout

```
.
├── main.go            # HTTP server, routing, OpenAI ⇄ provider translation
├── assets.go          # Embedded static assets (landing page & docs)
├── sessions.go        # In-memory session manager (opt-in via session_id)
├── middleware/        # Model directive registry & implementations
│   ├── middleware.go  #   registry, parsing, dispatch
│   ├── truncate.go    #   :truncate(target, limit)
│   ├── warp_tools.go  #   :warp_tools(top_k) dynamic tool retrieval & execution
│   └── semantic_search.go # fast local vector embedding search
├── providers/         # LLM provider implementations
│   ├── provider.go    #   shared types & Provider interface
│   ├── google.go      #   Gemini REST + Live WebSocket
│   └── deepseek.go    #   DeepSeek REST API (OpenAI format)
├── assets/            # Landing page source (embedded at build time)
├── config.example.json
├── Dockerfile
├── docker-compose.yml
└── build.sh
```

---

## 🤝 Contributing

We love contributions! ✨

1. **Fork** the repository.
2. **Create** your feature branch (`git checkout -b feature/AmazingFeature`).
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`).
4. **Push** to the branch (`git push origin feature/AmazingFeature`).
5. **Open** a Pull Request.

Please ensure your code follows `go fmt`. Thank you for making WarpGate better! ⭐
