# ⚡ Ollama-One (Google AI Proxy)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Go Report Card](https://goreportcard.com/badge/github.com/Neuxbane/Ollama-One)](https://goreportcard.com/report/github.com/Neuxbane/Ollama-One) [![Build State](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

> **High-Performance, Stateless AI Proxy**: Connect **VS Code Copilot** and **OpenAI-compatible clients** to Google Gemini REST and Gemini WebSocket Live streaming with native Tool Calling, Vision, and Extended Thinking support.

---

## 🎯 Key Capabilities

- 🛡️ **100% Stateless by Default**: No `config.json` required! Provide your Google Gemini API key directly per request via `Authorization: Bearer <key>`, `x-api-key`, or query parameters.
- 🧠 **Extended Thinking / Reasoning Effort**: Native support for `low`, `medium`, and `high` reasoning budgets directly in the VS Code Copilot chat bar.
- 🛠️ **Autonomous Agent Tool Calling**: Full tool-calling compatibility for VS Code Copilot Agent Mode (`read_file`, `replace_string_in_file`, terminal execution, etc.) with active connection reuse.
- 👁️ **Multimodal Vision**: Seamless image and screenshot analysis.
- ⚡ **Ultra-Low Latency Live Streaming**: Google Gemini Live bidirectional WebSocket streaming via `google/ws/<model>`.
- 📦 **One-Command Build & Docker Archive**: `./build.sh` builds the binary, creates a clean stateless Docker image, and exports `ollama-one-docker.tar.gz`.

---

## 🌐 Model Routing

- **`google/ws/<model>`** $\rightarrow$ Routes strictly to **Google Gemini WebSocket Live (Bidi)**
  - `google/ws/gemini-3.8-live-extended-thinking`
  - `google/ws/gemini-3.8-live`
  - `google/ws/gemini-3.1-flash-live-preview`
  - `google/ws/gemini-2.5-flash-native-audio-latest`
- **`google/<model>`** (or `<model>`) $\rightarrow$ Routes to **Google Gemini REST API**
  - `google/gemini-2.5-pro`
  - `google/gemini-2.5-flash`
  - `google/gemini-2.0-flash`

---

## 🚀 Quick Start

### Option 1: Run with Docker Compose (Stateless, Recommended)

Run directly without any configuration files:
```bash
docker compose up -d --build
```
The proxy will start on `http://127.0.0.1:11434`.

---

### Option 2: Build & Export Docker Archive (`.tar.gz`)

To build the Go binary and produce a portable, compressed Docker archive:
```bash
./build.sh
```

This generates:
- **`ollama-one`**: Native standalone executable
- **`ollama-one-docker.tar.gz`**: Portable stateless Docker image (~9.5 MB)

#### Load and run on any server/machine:
```bash
# 1. Load the archive
docker load < ollama-one-docker.tar.gz

# 2. Run the container statelessly
docker run -d -p 11434:11434 --name ollama-one ollama-one:latest
```

> [!TIP]
> Need to skip the Docker step during development? Run `./build.sh --no-docker`.

---

### Option 3: Optional Server-Wide Config File

If you prefer to configure a default fallback API key on the server (instead of providing it per request), create `config.json` in the root directory (see [`config.example.json`](config.example.json)):

```json
[
  {
    "type": "google",
    "key": "AIzaSy_YOUR_GOOGLE_GEMINI_API_KEY_HERE"
  }
]
```
You can also set the `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variable.

---

## 🔑 Authentication Options (Stateless)

When running statelessly, clients can pass the Gemini API key in any of the following standard ways:

1. **Standard `Authorization` header**:
   ```http
   Authorization: Bearer AIzaSy...
   ```
2. **`x-api-key` header** (Anthropic / Ollama style):
   ```http
   x-api-key: AIzaSy...
   ```
3. **`X-Goog-Api-Key` header** (Google Cloud style):
   ```http
   X-Goog-Api-Key: AIzaSy...
   ```
4. **URL Query Parameter**:
   ```http
   POST /v1/chat/completions?key=AIzaSy...
   ```
5. **JSON Body**:
   ```json
   {
     "model": "google/ws/gemini-3.8-live-extended-thinking",
     "api_key": "AIzaSy...",
     "messages": [...]
   }
   ```

---

## 🛠️ Step-by-Step: Connect to VS Code Copilot

### 📍 Step 1: Open Model Management
In the Copilot Chat panel, click the model selection dropdown at the bottom, then click **Manage Models...**.

![Step 1 - Manage Models](assets/step1.png)

---

### 📍 Step 2: Open Language Models Configuration (JSON)
In the top right corner of the Language Models settings tab, click the **Open Language Models (JSON)** file icon.

![Step 2 - Open Language Models JSON](assets/step2.png)

---

### 📍 Step 3: Add Custom Endpoint Configuration
Paste the configuration into `chatLanguageModels.json` (or add your Gemini API key in the `apiKey` field):

![Step 3 - Configuration Overview](assets/step3.png)

```json
{
  "name": "Custom Endpoint",
  "vendor": "customendpoint",
  "apiType": "chat-completions",
  "models": [
    {
      "id": "google/ws/gemini-3.1-flash-live-preview",
      "name": "Gemini 3.1 Flash Live (Recommended for Agent)",
      "url": "http://127.0.0.1:11434/v1/chat/completions",
      "toolCalling": true,
      "vision": true,
      "maxInputTokens": 128000,
      "maxOutputTokens": 16000
    },
    {
      "id": "google/ws/gemini-3.8-live-extended-thinking",
      "name": "Gemini 3.8 Flash Live Extended Thinking",
      "url": "http://127.0.0.1:11434/v1/chat/completions",
      "toolCalling": true,
      "vision": true,
      "supportsReasoningEffort": [
        "low",
        "medium",
        "high"
      ],
      "reasoningEffortFormat": "chat-completions",
      "maxInputTokens": 128000,
      "maxOutputTokens": 16000
    },
    {
      "id": "google/gemini-2.0-flash",
      "name": "Gemini 2.0 Flash (REST)",
      "url": "http://127.0.0.1:11434/v1/chat/completions",
      "toolCalling": true,
      "vision": true,
      "maxInputTokens": 128000,
      "maxOutputTokens": 16000
    }
  ]
}
```

---

## 🧠 Extended Thinking Setup

To enable reasoning effort controls (`low`, `medium`, `high`) directly in your VS Code chat input bar:

![Extended Thinking Configuration](assets/extended_thinking_config.png)

Ensure your model configuration contains these two attributes:
```json
"supportsReasoningEffort": [
  "low",
  "medium",
  "high"
],
"reasoningEffortFormat": "chat-completions"
```

---

## 🤖 VS Code Copilot Agent in Action

Here is the proxy powering the **VS Code Copilot Autonomous Agent**, reading workspace files, invoking tools, and analyzing codebases in real time:

![VS Code Copilot Agent in Action](assets/chatScreenshoot.png)

---

## 🧩 Model Middleware & Function Directives

Ollama-One supports dynamic model middlewares formatted as `:function(args)` directly in your model ID.

### Truncation Directive (`:truncate`)
Enforce maximum character limits on prompt input and tool/function responses to prevent token overflow and optimize context window usage:

```json
{
  "id": "google/ws/gemini-3.1-flash-live-preview:truncate(all, 4000)",
  "name": "Gemini 3.1 Flash Live (Truncate 4k)",
  "url": "http://127.0.0.1:11434/v1/chat/completions"
}
```

#### Available Syntax:
- **`:truncate(all, 4000)`**: Truncates all inputs, system prompts, and function/tool responses to a maximum of 4000 characters.
- **`:truncate(tools, 2000)`**: Truncates only function/tool responses and arguments to 2000 characters.
- **`:truncate(user, 1000)`**: Truncates only user messages.
- **`:truncate(4000)`**: Shorthand for `:truncate(all, 4000)`.

Truncation is unicode/rune-safe and appends an informative marker (e.g., `... [truncated N chars]`) while ensuring the total character count strictly does not exceed the limit.

#### Extensible Architecture:
Middlewares live in the specialized `middleware/` folder (e.g., [`middleware/truncate.go`](middleware/truncate.go)). New directives can be added simply by defining a `ModifierFunc` and registering it via `Register(name, fn)` in `init()`.

---

## 🤝 Contributing

We love contributions! ✨ Here's how to help:

1.  **Fork** the repository.
2.  **Create** your feature branch (`git checkout -b feature/AmazingFeature`).
3.  **Commit** your changes (`git commit -m 'Add some AmazingFeature'`).
4.  **Push** to the branch (`git push origin feature/AmazingFeature`).
5.  **Open** a Pull Request.

Thank you for making Ollama-One better! ⭐ Please ensure your code follows `go fmt`.
