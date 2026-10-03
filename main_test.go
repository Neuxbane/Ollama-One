package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
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
			ID:           "google/gemini-2.0-flash",
			Name:         "Gemini 2.0 Flash",
			ContextSize:  128000,
			Capabilities: []string{"vision", "tools"},
		},
		{
			ID:           "google/ws/gemini-3.1-flash-live-preview",
			Name:         "Gemini 3.1 Flash Live Preview (WebSocket)",
			ContextSize:  128000,
			Capabilities: []string{"vision", "tools", "audio"},
		},
	}, nil
}

func (m *MockProvider) ListModelsWithKey(ctx context.Context, apiKey string) ([]providers.ModelInfo, error) {
	return m.ListModels(ctx)
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
	parts := parseOpenAIMessageParts("Hello world", nil)
	if len(parts) != 1 || parts[0].Text != "Hello world" {
		t.Fatalf("Unexpected parts for string content: %+v", parts)
	}

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

func TestCleanUserPrompt(t *testing.T) {
	// Case 1: Copilot reminder instructions dump with userRequest
	rawInput := `<reminderInstructions>
It is much faster to edit using the replace_string_in_file tool. Prefer the replace_string_in_file tool for making edits and only fall back to insert_edit_into_file if it fails.
</reminderInstructions>
<userRequest>
can you read the file inside
</userRequest>`

	cleaned := cleanUserPrompt(rawInput)
	expected := "can you read the file inside"
	if cleaned != expected {
		t.Errorf("Expected %q, got %q", expected, cleaned)
	}

	// Case 2: Simple userRequest with "ok"
	rawOk := `<userRequest>
ok
</userRequest>`
	cleanedOk := cleanUserPrompt(rawOk)
	if cleanedOk != "ok" {
		t.Errorf("Expected 'ok', got %q", cleanedOk)
	}

	// Case 3: Regular text without tags
	regular := "hello how are you"
	if cleanUserPrompt(regular) != regular {
		t.Errorf("Expected regular text preserved, got %q", cleanUserPrompt(regular))
	}
}

func TestChatCompletionsVision(t *testing.T) {
	mock := &MockProvider{
		Resp: &providers.CompletionResponse{Content: "That is a red pixel."},
	}
	googleProvider = mock

	pngData := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	dataURI := "data:image/png;base64," + pngData

	reqBody := map[string]any{
		"model": "google/gemini-2.0-flash",
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

	handleChatCompletions(rec, httpReq)

	if rec.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

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

	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse response JSON: %v", err)
	}
	choices := resp["choices"].([]any)
	firstChoice := choices[0].(map[string]any)
	message := firstChoice["message"].(map[string]any)
	if message["content"] != "That is a red pixel." {
		t.Errorf("Unexpected content: %v", message["content"])
	}
}

func TestChatCompletionsTools(t *testing.T) {
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
	googleProvider = mock

	reqBody := map[string]any{
		"model": "google/gemini-2.0-flash",
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

	handleChatCompletions(rec, httpReq)

	if rec.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	if len(mock.LastReq.Tools) == 0 {
		t.Fatalf("Provider did not receive tool definitions")
	}
	if mock.LastReq.Tools[0].Name != "get_weather" {
		t.Errorf("Expected tool get_weather, got %s", mock.LastReq.Tools[0].Name)
	}

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

func TestChatCompletionsStreaming(t *testing.T) {
	mock := &MockProvider{
		Resp: &providers.CompletionResponse{
			Content: "Hello world!",
		},
	}
	googleProvider = mock

	reqBody := map[string]any{
		"model": "google/gemini-2.0-flash",
		"messages": []any{
			map[string]any{"role": "user", "content": "Hi"},
		},
		"stream": true,
	}

	bodyBytes, _ := json.Marshal(reqBody)
	httpReq := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(bodyBytes))
	rec := httptest.NewRecorder()

	handleChatCompletions(rec, httpReq)

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

func TestModelsEndpoint(t *testing.T) {
	mock := &MockProvider{}
	googleProvider = mock

	httpReq := httptest.NewRequest("GET", "/v1/models", nil)
	rec := httptest.NewRecorder()
	handleModels(rec, httpReq)

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

	httpReq2 := httptest.NewRequest("GET", "/v1/models/google/gemini-2.0-flash", nil)
	rec2 := httptest.NewRecorder()
	handleModels(rec2, httpReq2)

	if rec2.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d", rec2.Code)
	}
	var modelResp map[string]any
	if err := json.Unmarshal(rec2.Body.Bytes(), &modelResp); err != nil {
		t.Fatalf("Failed to parse single model JSON: %v", err)
	}
	if modelResp["id"] != "google/gemini-2.0-flash" {
		t.Errorf("Expected id google/gemini-2.0-flash, got %v", modelResp["id"])
	}
}

func TestChatCompletionsMultiTurnToolResult(t *testing.T) {
	mock := &MockProvider{
		Resp: &providers.CompletionResponse{
			Content: "The weather in Tokyo is 18°C and sunny.",
		},
	}
	googleProvider = mock

	reqBody := map[string]any{
		"model": "google/gemini-2.0-flash",
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

	handleChatCompletions(rec, httpReq)

	if rec.Code != http.StatusOK {
		t.Fatalf("Expected status 200, got %d: %s", rec.Code, rec.Body.String())
	}

	if mock.LastReq == nil || len(mock.LastReq.Messages) != 3 {
		t.Fatalf("Expected 3 messages in provider request, got %v", mock.LastReq)
	}

	asstMsg := mock.LastReq.Messages[1]
	if asstMsg.Role != "assistant" || len(asstMsg.ToolCalls) != 1 {
		t.Errorf("Expected assistant message with 1 tool call, got %+v", asstMsg)
	}
	if asstMsg.ToolCalls[0].ID != "call_abc123" {
		t.Errorf("Expected tool call ID call_abc123, got %s", asstMsg.ToolCalls[0].ID)
	}

	toolMsg := mock.LastReq.Messages[2]
	if toolMsg.Role != "tool" || toolMsg.ToolCallID != "call_abc123" || toolMsg.Name != "get_weather" {
		t.Errorf("Expected tool message with ID call_abc123 and name get_weather, got %+v", toolMsg)
	}
	if len(toolMsg.Content) == 0 || toolMsg.Content[0].Text != `{"temperature": "18C", "condition": "Sunny"}` {
		t.Errorf("Unexpected tool message content: %+v", toolMsg.Content)
	}
}

func TestExtractAPIKey(t *testing.T) {
	// 1. Authorization: Bearer
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer AIzaSy_from_bearer")
	if key := extractAPIKey(req, nil); key != "AIzaSy_from_bearer" {
		t.Errorf("Expected AIzaSy_from_bearer, got %q", key)
	}

	// 2. x-api-key
	req = httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("x-api-key", "AIzaSy_from_x_api_key")
	if key := extractAPIKey(req, nil); key != "AIzaSy_from_x_api_key" {
		t.Errorf("Expected AIzaSy_from_x_api_key, got %q", key)
	}

	// 3. X-Goog-Api-Key
	req = httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("X-Goog-Api-Key", "AIzaSy_from_goog_header")
	if key := extractAPIKey(req, nil); key != "AIzaSy_from_goog_header" {
		t.Errorf("Expected AIzaSy_from_goog_header, got %q", key)
	}

	// 4. Query param ?key=
	req = httptest.NewRequest("POST", "/v1/chat/completions?key=AIzaSy_from_query", nil)
	if key := extractAPIKey(req, nil); key != "AIzaSy_from_query" {
		t.Errorf("Expected AIzaSy_from_query, got %q", key)
	}

	// 5. Body field api_key
	req = httptest.NewRequest("POST", "/v1/chat/completions", nil)
	body := &OpenAIChatCompletionRequest{APIKey: "AIzaSy_from_body"}
	if key := extractAPIKey(req, body); key != "AIzaSy_from_body" {
		t.Errorf("Expected AIzaSy_from_body, got %q", key)
	}

	// 6. Dummy / placeholder tokens ignored
	req = httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer dummy")
	if key := extractAPIKey(req, nil); key != "" {
		t.Errorf("Expected dummy to be rejected, got %q", key)
	}
}

func TestChatCompletionsTruncateDirective(t *testing.T) {
	mock := &MockProvider{}
	oldProvider := googleProvider
	googleProvider = mock
	defer func() { googleProvider = oldProvider }()

	bodyJSON := `{
		"model": "google/ws/gemini-3.1-flash-live-preview:truncate(all, 100)",
		"messages": [
			{"role": "user", "content": "` + strings.Repeat("X", 500) + `"},
			{"role": "tool", "name": "read_file", "tool_call_id": "call_123", "content": "` + strings.Repeat("Y", 600) + `"}
		]
	}`

	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(bodyJSON))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handleChatCompletions(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d: %s", w.Code, w.Body.String())
	}

	if mock.LastReq == nil {
		t.Fatalf("Expected mock provider to receive request")
	}

	// 1. Base model must be cleaned
	if mock.LastReq.Model != "google/ws/gemini-3.1-flash-live-preview" {
		t.Errorf("Expected cleaned base model, got %q", mock.LastReq.Model)
	}

	// 2. User input truncated to <= 100 runes
	userText := mock.LastReq.Messages[0].Content[0].Text
	if len([]rune(userText)) > 100 {
		t.Errorf("Expected user message <= 100 runes, got %d", len([]rune(userText)))
	}

	// 3. Function response truncated to <= 100 runes
	funcText := mock.LastReq.Messages[1].Content[0].Text
	if len([]rune(funcText)) > 100 {
		t.Errorf("Expected function response <= 100 runes, got %d", len([]rune(funcText)))
	}
	if !strings.Contains(funcText, "[truncated") {
		t.Errorf("Expected truncation marker in function response: %s", funcText)
	}

	// 4. Response JSON maintains requested model ID for client compatibility
	var respMap map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &respMap); err == nil {
		if respModel, ok := respMap["model"].(string); ok {
			if respModel != "google/ws/gemini-3.1-flash-live-preview:truncate(all, 100)" {
				t.Errorf("Expected client response model to match requested model, got %q", respModel)
			}
		}
	}
}

func TestDeepSeekRoutingAndCompletions(t *testing.T) {
	mockDS := &MockProvider{
		Resp: &providers.CompletionResponse{
			Content: "Response from DeepSeek",
			Thought: "Deep reasoning...",
		},
	}
	oldDS := deepseekProvider
	deepseekProvider = mockDS
	defer func() { deepseekProvider = oldDS }()

	bodyJSON := `{
		"model": "deepseek/deepseek-chat:truncate(all, 200)",
		"messages": [
			{"role": "user", "content": "Hello DeepSeek"}
		]
	}`

	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(bodyJSON))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handleChatCompletions(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d: %s", w.Code, w.Body.String())
	}

	if mockDS.LastReq == nil {
		t.Fatalf("Expected deepseekProvider to receive the request")
	}

	if mockDS.LastReq.Model != "deepseek/deepseek-chat" {
		t.Errorf("Expected cleaned base model deepseek/deepseek-chat, got %q", mockDS.LastReq.Model)
	}

	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse JSON response: %v", err)
	}

	choices, ok := resp["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatalf("Invalid or missing choices in response")
	}
	choice := choices[0].(map[string]any)
	msg := choice["message"].(map[string]any)

	if msg["content"] != "Response from DeepSeek" {
		t.Errorf("Expected content 'Response from DeepSeek', got %v", msg["content"])
	}
	if msg["reasoning_content"] != "Deep reasoning..." {
		t.Errorf("Expected reasoning_content 'Deep reasoning...', got %v", msg["reasoning_content"])
	}
	if resp["system_fingerprint"] != "fp_deepseek_proxy" {
		t.Errorf("Expected system_fingerprint fp_deepseek_proxy, got %v", resp["system_fingerprint"])
	}
}

func TestDeepSeekModelsEndpoint(t *testing.T) {
	// Single model query
	req := httptest.NewRequest("GET", "/v1/models/deepseek-chat", nil)
	w := httptest.NewRecorder()
	handleModels(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d", w.Code)
	}

	var singleModel map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &singleModel); err != nil {
		t.Fatalf("Failed to decode model JSON: %v", err)
	}
	if singleModel["id"] != "deepseek-chat" || singleModel["owned_by"] != "deepseek" {
		t.Errorf("Unexpected single model response: %+v", singleModel)
	}

	// Models list
	reqList := httptest.NewRequest("GET", "/v1/models", nil)
	wList := httptest.NewRecorder()
	handleModels(wList, reqList)

	if wList.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d", wList.Code)
	}

	var listResp struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	if err := json.Unmarshal(wList.Body.Bytes(), &listResp); err != nil {
		t.Fatalf("Failed to decode models list: %v", err)
	}

	hasDeepSeek := false
	hasGoogle := false
	for _, m := range listResp.Data {
		if strings.Contains(m.ID, "deepseek") && m.OwnedBy == "deepseek" {
			hasDeepSeek = true
		}
		if strings.Contains(m.ID, "gemini") && m.OwnedBy == "google" {
			hasGoogle = true
		}
	}

	if !hasDeepSeek {
		t.Errorf("Expected deepseek models in /v1/models output")
	}
	if !hasGoogle {
		t.Errorf("Expected google models in /v1/models output")
	}
}

type mockWarpServerProvider struct {
	turn int
}

func (m *mockWarpServerProvider) ListModels(ctx context.Context) ([]providers.ModelInfo, error) {
	return nil, nil
}
func (m *mockWarpServerProvider) ListModelsWithKey(ctx context.Context, apiKey string) ([]providers.ModelInfo, error) {
	return nil, nil
}
func (m *mockWarpServerProvider) Chat(ctx context.Context, req *providers.CompletionRequest, onChunk func(*providers.CompletionResponse)) (*providers.CompletionResponse, error) {
	m.turn++
	if m.turn == 1 {
		return &providers.CompletionResponse{
			ToolCalls: []providers.ToolCall{
				{
					ID:   "call_search_1",
					Type: "function",
					Function: providers.FunctionCall{
						Name:      "searchTools",
						Arguments: `{"query": "file editor"}`,
					},
				},
			},
		}, nil
	}
	return &providers.CompletionResponse{
		ToolCalls: []providers.ToolCall{
			{
				ID:   "call_exec_1",
				Type: "function",
				Function: providers.FunctionCall{
					Name:      "execute",
					Arguments: `{"tools": [{"name": "write_file", "arguments": {"path": "test.txt", "content": "hello"}}]}`,
				},
			},
		},
	}, nil
}

func TestChatCompletionsWarpToolsDirective(t *testing.T) {
	mock := &mockWarpServerProvider{}
	oldProvider := googleProvider
	googleProvider = mock
	defer func() { googleProvider = oldProvider }()

	var tools []map[string]any
	tools = append(tools, map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        "write_file",
			"description": "Write contents to a file on disk",
			"parameters":  map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}},
		},
	})
	for i := 2; i <= 15; i++ {
		tools = append(tools, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        fmt.Sprintf("other_tool_%d", i),
				"description": fmt.Sprintf("Unrelated tool %d", i),
			},
		})
	}

	reqBody := map[string]any{
		"model": "gemini-2.0-flash:warp_tools(top_k=5)",
		"messages": []map[string]any{
			{"role": "user", "content": "Write to test.txt"},
		},
		"tools": tools,
	}

	bodyJSON, _ := json.Marshal(reqBody)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBuffer(bodyJSON))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handleChatCompletions(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("Expected 200 OK, got %d: %s", w.Code, w.Body.String())
	}

	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("Failed to parse JSON response: %v", err)
	}

	choices := resp["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	toolCalls, ok := msg["tool_calls"].([]any)
	if !ok || len(toolCalls) != 1 {
		t.Fatalf("Expected 1 unwrapped tool call, got %+v", msg)
	}

	tc := toolCalls[0].(map[string]any)
	fn := tc["function"].(map[string]any)
	if fn["name"] != "write_file" {
		t.Errorf("Expected unwrapped tool name write_file, got %v", fn["name"])
	}
	if !strings.Contains(fmt.Sprint(fn["arguments"]), "test.txt") {
		t.Errorf("Expected test.txt in arguments, got %v", fn["arguments"])
	}
}



