# ⚡ Ollama-One

> **High-Performance AI Proxy**: Connect **VS Code Copilot**, **Ollama**, and **OpenAI-compatible clients** to Google Gemini & Gemini Live with native Tool Calling, Vision, and Extended Thinking support.

---

## 🎯 Key Capabilities

- 🧠 **Extended Thinking / Reasoning Effort**: Native support for `low`, `medium`, and `high` reasoning budgets directly in VS Code Copilot.
- 🛠️ **Autonomous Agent Tool Calling**: Full tool-calling compatibility for VS Code Copilot Agent Mode (`read_file`, `replace_string_in_file`, terminal commands, etc.).
- 👁️ **Multimodal Vision**: Seamless image and screenshot analysis.
- ⚡ **Ultra-Low Latency Live Streaming**: WebSocket bidirectional streaming via Gemini Live.
- 🔄 **Universal Compatibility**: Serves both `/v1/chat/completions` (OpenAI format) and `/api/chat` (Ollama format).

---

## 🚀 Quick Start

### 1️⃣ Configure Your API Key

Create `config.json` in the root directory (or copy from `config.example.json`):

```json
[
  {
    "type": "gemini",
    "key": "AIzaSy_YOUR_GEMINI_API_KEY_HERE"
  }
]
```

> [!TIP]
> A single `"type": "gemini"` key automatically powers **both** standard Gemini models and Gemini Live models!

---

### 2️⃣ Start Ollama-One

Choose your preferred way to run:

#### 🐳 Option A: Run with Docker Compose (Recommended)
Docker is configured with `restart: unless-stopped`, so it will automatically start on boot:
```bash
docker compose up -d --build
```

#### 💻 Option B: Run Locally
```bash
./build.sh
./ollama-one
```

Ollama-One will start listening on `http://127.0.0.1:11434`.

---

## 🛠️ Step-by-Step: Connect to VS Code Copilot

Follow these 3 simple steps to integrate Ollama-One with VS Code Copilot:

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

![Step 3 - Add Custom Endpoint](assets/step3.png)

```json
{
  "name": "Custom Endpoint",
  "vendor": "customendpoint",
  "apiType": "chat-completions",
  "models": [
    {
      "id": "gemini-3.1-flash-live-preview",
      "name": "Gemini 3.1 Flash Live (Recommended for Agent)",
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
      "id": "gemini-3.8-live-extended-thinking",
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
    }
  ]
}
```

---

## 🧠 Extended Thinking Setup

To enable reasoning effort controls (`low`, `medium`, `high`) directly in your VS Code chat input bar:

![Extended Thinking Config](assets/extended_thinking_config.png)

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

Here is Ollama-One powering the **VS Code Copilot Autonomous Agent**, actively reading workspace files, invoking tools, and analyzing codebases:

![VS Code Copilot Agent in Action](assets/chatScreenshoot.png)
