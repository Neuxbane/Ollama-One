# ⚡ Ollama-One (Google AI Proxy)

> **High-Performance AI Proxy**: Connect **VS Code Copilot** and **OpenAI-compatible clients** to Google Gemini REST and Gemini WebSocket Live streaming with native Tool Calling, Vision, and Extended Thinking support.

---

## 🎯 Key Capabilities

- 🧠 **Extended Thinking / Reasoning Effort**: Native support for `low`, `medium`, and `high` reasoning budgets directly in VS Code Copilot.
- 🛠️ **Autonomous Agent Tool Calling**: Full tool-calling compatibility for VS Code Copilot Agent Mode (`read_file`, `replace_string_in_file`, terminal commands, etc.).
- 👁️ **Multimodal Vision**: Seamless image and screenshot analysis.
- ⚡ **Ultra-Low Latency Live Streaming**: WebSocket bidirectional streaming via `google/ws/<model>`.
- 🌐 **Clean Model Routing**:
  - `google/<model>` (or `<model>`) -> Google Gemini REST API
  - `google/ws/<model>` -> Google Gemini Live WebSocket Bidi streaming

---

## 🚀 Quick Start

### 1️⃣ Configure Your API Key

Create `config.json` in the root directory (or copy from `config.example.json`):

```json
[
  {
    "type": "google",
    "key": "AIzaSy_YOUR_GOOGLE_GEMINI_API_KEY_HERE"
  }
]
```

> [!TIP]
> A single `"type": "google"` (or `"type": "gemini"`) key powers **both** standard Gemini models and Gemini Live WebSocket models! You can also set `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variables.

---

### 2️⃣ Start the Proxy

Choose your preferred way to run:

#### 🐳 Option A: Run with Docker Compose (Recommended)
Docker is configured with `restart: unless-stopped`, so it will automatically start on boot:
```bash
docker compose up -d --build
```

#### 💻 Option B: Run Locally
```bash
./build.sh
```

The proxy will start listening on `http://127.0.0.1:11434`.

---

## 🛠️ Step-by-Step: Connect to VS Code Copilot

Follow these simple steps to integrate the proxy with VS Code Copilot:

### 📍 Step 1: Open Model Management
In the Copilot Chat panel, click the model selection dropdown at the bottom, then click **Manage Models...**.

![Step 1 - Manage Models](assets/step1.png)

---

### 📍 Step 2: Open Language Models Configuration (JSON)
In the top right corner of the Language Models settings tab, click the **Open Language Models (JSON)** file icon.

![Step 2 - Open Language Models JSON](assets/step2.png)

---

### 📍 Step 3: Add Custom Endpoint Configuration
Paste the custom endpoint configuration into the `chatLanguageModels.json` file:

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
,,b nbn snb nb nb ddshha,ba,,

Ensure your model configuration contains the following two fields:
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

Here is the proxy powering the **VS Code Copilot Autonomous Agent**, actively reading workspace files, invoking tools, and analyzing codebases:

![VS Code Copilot Agent in Action](assets/chatScreenshoot.png)
