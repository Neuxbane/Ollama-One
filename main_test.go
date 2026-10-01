package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Neuxbane/Ollama-One/providers"
)

// MockProvider implements providers.Provider for testing
type MockProvider struct {
	LastReq *providers.CompletionRequest
	Resp    *providers.CompletionResponse
}

func (m *MockProvider) ListModels(ctx context.Context) ([]providers.ModelInfo, error) {
	return []providers.ModelInfo{
		{
			ID:           "mock-model",
			Name:         "Mock Model",
			ContextSize:  128000,
			Capabilities: []string{"vision", "tools"},
		},
	}, nil
}

func (m *MockProvider) Chat(ctx context.Context, req *providers.CompletionRequest, onChunk func(*providers.CompletionResponse)) (*providers.CompletionResponse, error) {
	m.LastReq = req
	if req.Stream && onChunk != nil {
		if m.Resp != nil {
			if m.Resp.Content != "" {
				onChunk(&providers.CompletionResponse{Content: m.Resp.Content})
			}
			if len(m.Resp.ToolCalls) > 0 {
				onChunk(&providers.CompletionResponse{ToolCalls: m.Resp.ToolCalls})
			}
		} else {
			onChunk(&providers.CompletionResponse{Content: "streaming mock reply"})
		}
		return m.Resp, nil
	}
	if m.Resp != nil {
		return m.Resp, nil
	}
	return &providers.CompletionResponse{Content: "mock reply"}, nil
}

func TestParseImageURLToContentPart(t *testing.T) {
	// 1. Data URI with PNG header
	pngData := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	dataURI := "data:image/png;base64," + pngData

	part, err := parseImageURLToContentPart(dataURI)
	if err != nil {
		t.Fatalf("Failed to parse data URI: %v", err)
	}
	if part.Type != providers.ContentTypeImage {
		t.Errorf("Expected ContentTypeImage, got %s", part.Type)
	}
	if part.MimeType != "image/png" {
		t.Errorf("Expected image/png, got %s", part.MimeType)
	}
	expectedBytes, _ := base64.StdEncoding.DecodeString(pngData)
	if !bytes.Equal(part.Data, expectedBytes) {
		t.Errorf("Decoded image bytes do not match expected")
	}

	// 2. Raw base64 string
	partRaw, err := parseImageURLToContentPart(pngData)
	if err != nil {
		t.Fatalf("Failed to parse raw base64: %v", err)
	}
	if partRaw.Type != providers.ContentTypeImage {
		t.Errorf("Expected ContentTypeImage, got %s", partRaw.Type)
	}
	if !strings.HasPrefix(partRaw.MimeType, "image/") {
		t.Errorf("Expected image/* mime, got %s", partRaw.MimeType)
	}
}

func TestParseOpenAIMessageParts(t *testing.T) {
	// String content
	parts := parseOpenAIMessageParts("Hello world", nil)
	if len(parts) != 1 || parts[0].Text != "Hello world" {
		t.Fatalf("Unexpected parts for string content: %+v", parts)
	}

	// Array of text + image_url
	pngData := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	dataURI := "data:image/png;base64," + pngData

	contentArray := []any{
		map[string]any{"type": "text", "text": "Describe this image:"},
		map[string]any{
			"type": "image_url",
			"image_url": map[string]any{
				"url": dataURI,
			},
		},
	}

	parts2 := parseOpenAIMessageParts(contentArray, nil)
	if len(parts2) != 2 {
		t.Fatalf("Expected 2 parts, got %d", len(parts2))
	}
	if parts2[0].Type != providers.ContentTypeText || parts2[0].Text != "Describe this image:" {
		t.Errorf("Part 0 mismatch: %+v", parts2[0])
	}
	if parts2[1].Type != providers.ContentTypeImage || parts2[1].MimeType != "image/png" {
		t.Errorf("Part 1 mismatch: %+v", parts2[1])
	}
}

func TestOpenAIChatCompletionsVision(t *testing.T) {
	mock := &MockProvider{
		Resp: &providers.CompletionResponse{Content: "That is a red pixel."},
	}
	providersMap["mock"] = mock

	pngData := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	dataURI := "data:image/png;base64," + pngData

	reqBody := map[string]any{
		"model": "mock/model",
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "What is this?"},
					map[string]any{
						"type": "image_url",
						"image_url": map[string]any{
							"url": dataURI,
						},
					},
				},
			},
		},
		"stream": false,
	}

	bodyBytes, _ := json.Marshal(reqBody)
	httpReq := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(bodyBytes))
	rec := httptest.NewRecorder()

	handleOpenAIChatCompletions(rec, httpReq)

	if rec.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	// Verify the mock received the image part
	if mock.LastReq == nil || len(mock.LastReq.Messages) == 0 {
		t.Fatalf("Provider did not receive messages")
	}
	msg := mock.LastReq.Messages[0]
	if len(msg.Content) != 2 {
		t.Fatalf("Expected 2 parts in message, got %d", len(msg.Content))
	}
	if msg.Content[1].Type != providers.ContentTypeImage {
		t.Errorf("Expected ContentTypeImage for part 1, got %s", msg.Content[1].Type)
	}

	// Verify response structure
	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response JSON: %v", err)
	}
	if resp["object"] != "chat.completion" {
		t.Errorf("Expected object chat.completion, got %v", resp["object"])
	}
	choices := resp["choices"].([]any)
	firstChoice := choices[0].(map[string]any)
	message := firstChoice["message"].(map[string]any)
	if message["content"] != "That is a red pixel." {
		t.Errorf("Unexpected content: %v", message["content"])
	}
}

func TestOpenAIChatCompletionsTools(t *testing.T) {
	mock := &MockProvider{
		Resp: &providers.CompletionResponse{
			Content: "",
			ToolCalls: []providers.ToolCall{
				{
					ID:   "call_12345",
					Type: "function",
					Function: providers.FunctionCall{
						Name:      "get_weather",
						Arguments: `{"location":"Tokyo"}`,
					},
				},
			},
		},
	}
	providersMap["mock"] = mock

	reqBody := map[string]any{
		"model": "mock/model",
		"messages": []any{
			map[string]any{
				"role":    "user",
				"content": "What's the weather in Tokyo?",
			},
		},
		"tools": []any{
			map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        "get_weather",
					"description": "Get current weather",
					"parameters": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"location": map[string]any{"type": "string"},
						},
						"required": []string{"location"},
					},
				},
			},
		},
		"stream": false,
	}

	bodyBytes, _ := json.Marshal(reqBody)
	httpReq := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(bodyBytes))
	rec := httptest.NewRecorder()

	handleOpenAIChatCompletions(rec, httpReq)

	if rec.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	// Verify provider received tool definition
	if len(mock.LastReq.Tools) == 0 {
		t.Fatalf("Provider did not receive tool definitions")
	}
	if mock.LastReq.Tools[0].Name != "get_weather" {
		t.Errorf("Expected tool get_weather, got %s", mock.LastReq.Tools[0].Name)
	}

	// Verify response has tool_calls and finish_reason: "tool_calls"
	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response JSON: %v", err)
	}
	choices := resp["choices"].([]any)
	firstChoice := choices[0].(map[string]any)
	if firstChoice["finish_reason"] != "tool_calls" {
		t.Errorf("Expected finish_reason 'tool_calls', got %v", firstChoice["finish_reason"])
	}
	message := firstChoice["message"].(map[string]any)
	toolCalls, ok := message["tool_calls"].([]any)
	if !ok || len(toolCalls) == 0 {
		t.Fatalf("Expected tool_calls in message, got: %v", message)
	}
	tc0 := toolCalls[0].(map[string]any)
	if tc0["id"] != "call_12345" {
		t.Errorf("Expected id call_12345, got %v", tc0["id"])
	}
	fn := tc0["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("Expected function name get_weather, got %v", fn["name"])
	}
	if fn["arguments"] != `{"location":"Tokyo"}` {
		t.Errorf("Expected arguments string, got %v", fn["arguments"])
	}
}

func TestOpenAIChatCompletionsStreaming(t *testing.T) {
	mock := &MockProvider{
		Resp: &providers.CompletionResponse{
			Content: "Hello world!",
		},
	}
	providersMap["mock"] = mock

	reqBody := map[string]any{
		"model": "mock/model",
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
		"stream": true,
	}

	bodyBytes, _ := json.Marshal(reqBody)
	httpReq := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(bodyBytes))
	rec := httptest.NewRecorder()

	handleOpenAIChatCompletions(rec, httpReq)

	if rec.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	bodyStr := rec.Body.String()
	if !strings.Contains(bodyStr, "data: [DONE]") {
		t.Errorf("Streaming response did not contain [DONE]: %s", bodyStr)
	}
	if !strings.Contains(bodyStr, "chat.completion.chunk") {
		t.Errorf("Streaming response did not contain chat.completion.chunk: %s", bodyStr)
	}
}

func TestOpenAIModelsEndpoint(t *testing.T) {
	mock := &MockProvider{}
	providersMap["mock"] = mock

	// Test list models
	httpReq := httptest.NewRequest("GET", "/v1/models", nil)
	rec := httptest.NewRecorder()
	handleOpenAIModels(rec, httpReq)

	if rec.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d", rec.Code)
	}

	var listResp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &listResp); err != nil {
		t.Fatalf("Failed to parse models JSON: %v", err)
	}
	if listResp["object"] != "list" {
		t.Errorf("Expected object 'list', got %v", listResp["object"])
	}

	// Test single model
	httpReq2 := httptest.NewRequest("GET", "/v1/models/gpt-4o", nil)
	rec2 := httptest.NewRecorder()
	handleOpenAIModels(rec2, httpReq2)

	if rec2.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d", rec2.Code)
	}
	var modelResp map[string]any
	if err := json.Unmarshal(rec2.Body.Bytes(), &modelResp); err != nil {
		t.Fatalf("Failed to parse single model JSON: %v", err)
	}
	if modelResp["id"] != "gpt-4o" {
		t.Errorf("Expected id gpt-4o, got %v", modelResp["id"])
	}
}

func TestOpenAIChatCompletionsMultiTurnToolResult(t *testing.T) {
	mock := &MockProvider{
		Resp: &providers.CompletionResponse{
			Content: "The weather in Tokyo is 18°C and sunny.",
		},
	}
	providersMap["mock"] = mock

	reqBody := map[string]any{
		"model": "mock/model",
		"messages": []any{
			map[string]any{
				"role":    "user",
				"content": "What's the weather in Tokyo?",
			},
			map[string]any{
				"role":    "assistant",
				"content": nil,
				"tool_calls": []any{
					map[string]any{
						"id":   "call_abc123",
						"type": "function",
						"function": map[string]any{
							"name":      "get_weather",
							"arguments": `{"location":"Tokyo"}`,
						},
					},
				},
			},
			map[string]any{
				"role":         "tool",
				"tool_call_id": "call_abc123",
				"name":         "get_weather",
				"content":      `{"temperature": "18C", "condition": "Sunny"}`,
			},
		},
		"stream": false,
	}

	bodyBytes, _ := json.Marshal(reqBody)
	httpReq := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(bodyBytes))
	rec := httptest.NewRecorder()

	handleOpenAIChatCompletions(rec, httpReq)

	if rec.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	if mock.LastReq == nil || len(mock.LastReq.Messages) != 3 {
		t.Fatalf("Expected 3 messages in provider request, got %v", mock.LastReq)
	}

	// Message 1: assistant with tool calls
	asstMsg := mock.LastReq.Messages[1]
	if asstMsg.Role != "assistant" || len(asstMsg.ToolCalls) != 1 {
		t.Errorf("Expected assistant message with 1 tool call, got %+v", asstMsg)
	}
	if asstMsg.ToolCalls[0].ID != "call_abc123" {
		t.Errorf("Expected tool call ID call_abc123, got %s", asstMsg.ToolCalls[0].ID)
	}

	// Message 2: tool message
	toolMsg := mock.LastReq.Messages[2]
	if toolMsg.Role != "tool" || toolMsg.ToolCallID != "call_abc123" || toolMsg.Name != "get_weather" {
		t.Errorf("Expected tool message with ID call_abc123 and name get_weather, got %+v", toolMsg)
	}
	if len(toolMsg.Content) == 0 || toolMsg.Content[0].Text != `{"temperature": "18C", "condition": "Sunny"}` {
		t.Errorf("Unexpected tool message content: %+v", toolMsg.Content)
	}
}

func TestOllamaChatEndpointCompatibility(t *testing.T) {
	mock := &MockProvider{
		Resp: &providers.CompletionResponse{
			Content: "Hello from Ollama!",
		},
	}
	providersMap["mock"] = mock

	reqBody := OllamaChatRequest{
		Model: "mock/model",
		Messages: []OllamaMessage{
			{Role: "user", Content: "Hello"},
		},
		Stream: false,
	}

	bodyBytes, _ := json.Marshal(reqBody)
	httpReq := httptest.NewRequest("POST", "/api/chat", bytes.NewReader(bodyBytes))
	rec := httptest.NewRecorder()

	handleChat(rec, httpReq, &reqBody)

	if rec.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp OllamaChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse Ollama response: %v", err)
	}
	if resp.Message.Content != "Hello from Ollama!" {
		t.Errorf("Unexpected Ollama message content: %s", resp.Message.Content)
	}
}

func TestLiveModelRouting(t *testing.T) {
	liveMock := &MockProvider{
		Resp: &providers.CompletionResponse{
			Content: "Hello from Gemini Live!",
		},
	}
	geminiLiveProvider = &providers.GeminiLiveProvider{
		BaseProvider: providers.BaseProvider{APIKeys: []string{"dummy"}},
	}
	providersMap["gemini-live"] = liveMock

	// Test pattern match on "gemini-3.1-flash-live-preview"
	p, targetModel := getProvider("gemini-3.1-flash-live-preview")
	if p == nil {
		t.Fatalf("Expected provider for gemini-3.1-flash-live-preview, got nil")
	}
	if targetModel != "gemini-3.1-flash-live-preview" {
		t.Errorf("Expected target model 'gemini-3.1-flash-live-preview', got '%s'", targetModel)
	}

	// Test pattern match on "gemini-3.8-live-extended-thinking"
	p3, targetModel3 := getProvider("gemini-3.8-live-extended-thinking")
	if p3 == nil {
		t.Fatalf("Expected provider for gemini-3.8-live-extended-thinking, got nil")
	}
	if targetModel3 != "gemini-3.8-live-extended-thinking" {
		t.Errorf("Expected target model 'gemini-3.8-live-extended-thinking', got '%s'", targetModel3)
	}

	// Test pattern match on "gemini-3.8-live"
	p4, targetModel4 := getProvider("gemini-3.8-live")
	if p4 == nil {
		t.Fatalf("Expected provider for gemini-3.8-live, got nil")
	}
	if targetModel4 != "gemini-3.8-live" {
		t.Errorf("Expected target model 'gemini-3.8-live', got '%s'", targetModel4)
	}
}

