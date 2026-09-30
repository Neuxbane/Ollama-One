# Ollama-One

Ollama-One acts as a proxy connecting to models like Gemini, mimicking the Ollama API.

## Setup

Follow these steps to configure Ollama-One:

| Step 1 | Step 2 | Step 3 |
|---|---|---|
| ![Step 1](assets/step1.png) | ![Step 2](assets/step2.png) | ![Step 3](assets/step3.png) |

## Chat Interface

Here is a preview of the chat interface:

![Chat Screenshot](assets/chatScreenshoot.png)

## Extended Thinking Configuration

For VS Code Copilot or OpenAI-compatible custom endpoints with Extended Thinking support:

![Extended Thinking](assets/extended_thinking.png)

![Extended Thinking Config](assets/extended_thinking_config.png)

```json
{
  "name": "Custom Endpoint",
  "vendor": "customendpoint",
  "apiType": "chat-completions",
  "models": [
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
    },
    {
      "id": "gemini-3.8-live",
      "name": "Gemini 3.8 Flash Live",
      "url": "http://127.0.0.1:11434/v1/chat/completions",
      "toolCalling": true,
      "vision": true,
      "maxInputTokens": 128000,
      "maxOutputTokens": 16000
    },
    {
      "id": "gemini-3.1-flash-live-preview",
      "name": "Gemini 3.1 Flash Live Preview",
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

