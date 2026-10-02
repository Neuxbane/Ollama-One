package providers

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestGoogleProviderListModelsLive(t *testing.T) {
	p := NewGoogleProvider([]string{"test-api-key"})
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	foundLive := false
	for _, m := range models {
		if strings.HasPrefix(m.ID, "google/ws/") {
			foundLive = true
			break
		}
	}

	if !foundLive {
		t.Errorf("Expected google/ws/ in live models")
	}
}



func TestSanitizeGeminiSchema(t *testing.T) {
	input := map[string]any{
		"$schema":     "http://json-schema.org/draft-07/schema#",
		"title":       "ToolInput",
		"type":        "object",
		"description": "Some description",
		"properties": map[string]any{
			"path": map[string]any{
				"type":        "string",
				"description": "File path",
			},
			"count": map[string]any{
				"type": []any{"integer", "null"},
			},
		},
		"required": []any{"path", "non_existent_key"},
	}

	cleaned := sanitizeGeminiSchemaMap(input)

	if _, exists := cleaned["$schema"]; exists {
		t.Errorf("Expected $schema to be stripped")
	}
	if _, exists := cleaned["title"]; exists {
		t.Errorf("Expected title to be stripped")
	}
	if cleaned["type"] != "OBJECT" {
		t.Errorf("Expected type OBJECT, got %v", cleaned["type"])
	}

	props := cleaned["properties"].(map[string]any)
	pathProp := props["path"].(map[string]any)
	if pathProp["type"] != "STRING" {
		t.Errorf("Expected property type STRING, got %v", pathProp["type"])
	}

	req := cleaned["required"].([]any)
	if len(req) != 1 || req[0] != "path" {
		t.Errorf("Expected required to only contain valid property 'path', got %v", req)
	}
}

func TestFixToolCall(t *testing.T) {
	p := NewGoogleProvider([]string{"dummy"})

	availableTools := []Tool{
		{
			Type: "function",
			Functions: []Tool{
				{
					Name: "read_file",
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"filePath": map[string]any{"type": "string"},
						},
						"required": []any{"filePath"},
					},
				},
			},
		},
	}

	// Model returned "path" instead of "filePath"
	rawArgs := map[string]any{
		"path": "/workspace/main.go",
	}

	fixed, err := p.fixToolCall("read_file", rawArgs, availableTools)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if val, ok := fixed["filePath"].(string); !ok || val != "/workspace/main.go" {
		t.Errorf("Expected filePath='/workspace/main.go', got %v", fixed)
	}
}

func TestBuildGeminiContents(t *testing.T) {
	p := NewGoogleProvider([]string{"dummy"})

	messages := []Message{
		{
			Role: "user",
			Content: []ContentPart{
				{Type: ContentTypeText, Text: "can you scan this codebase"},
			},
		},
		{
			Role: "assistant",
			ToolCalls: []ToolCall{
				{
					ID:   "call_123",
					Type: "function",
					Function: FunctionCall{
						Name:      "read_file",
						Arguments: `{"filePath":"main.go"}`,
					},
				},
			},
		},
		{
			Role:       "tool",
			ToolCallID: "call_123",
			Content: []ContentPart{
				{Type: ContentTypeText, Text: "package main"},
			},
		},
	}

	// Live mode: turns are formatted as standardized JSON inside special tags <tool_calls> and <tool_responses>
	liveContents, sys := p.buildGeminiContents(messages, true)
	if len(sys) != 0 {
		t.Errorf("Expected 0 system instructions, got %d", len(sys))
	}
	if len(liveContents) != 3 {
		t.Fatalf("Expected 3 turns, got %d", len(liveContents))
	}
	if liveContents[0].Role != "user" || liveContents[0].Parts[0].Text != "can you scan this codebase" {
		t.Errorf("Turn 0 mismatch: %+v", liveContents[0])
	}
	if liveContents[1].Role != "model" || !strings.Contains(liveContents[1].Parts[0].Text, "<tool_calls>") || !strings.Contains(liveContents[1].Parts[0].Text, "read_file") {
		t.Errorf("Turn 1 mismatch: %+v", liveContents[1])
	}
	if liveContents[2].Role != "user" || !strings.Contains(liveContents[2].Parts[0].Text, "<tool_responses>") || !strings.Contains(liveContents[2].Parts[0].Text, "package main") {
		t.Errorf("Turn 2 mismatch: %+v", liveContents[2])
	}

	// REST mode: tool role becomes user with structured functionResponse part (Gemini 2.5/3.8 requirement)
	restContents, _ := p.buildGeminiContents(messages, false)
	if len(restContents) != 3 {
		t.Fatalf("Expected 3 turns, got %d", len(restContents))
	}
	if restContents[1].Role != "model" || restContents[1].Parts[0].FunctionCall == nil || restContents[1].Parts[0].FunctionCall.Name != "read_file" {
		t.Errorf("Turn 1 REST mismatch: %+v", restContents[1])
	}
	if restContents[2].Role != "user" || restContents[2].Parts[0].FunctionResponse == nil {
		t.Errorf("Turn 2 REST mismatch: %+v", restContents[2])
	}
}

func TestThoughtSignaturePreservation(t *testing.T) {
	p := NewGoogleProvider([]string{"dummy"})
	fakeSig := "ErcBCrQBAWkUfROD+4OQrSsolzVGjrE..."
	cacheThoughtSignature("call_test_123", "read_file", fakeSig)

	messages := []Message{
		{
			Role:    "user",
			Content: []ContentPart{{Type: ContentTypeText, Text: "read file"}},
		},
		{
			Role: "assistant",
			ToolCalls: []ToolCall{
				{
					ID:       "call_test_123",
					Type:     "function",
					Function: FunctionCall{Name: "read_file", Arguments: `{"filePath":"README.md"}`},
				},
			},
		},
		{
			Role:       "tool",
			Name:       "read_file",
			ToolCallID: "call_test_123",
			Content:    []ContentPart{{Type: ContentTypeText, Text: "file content 1"}},
		},
		{
			Role:       "tool",
			Name:       "read_file",
			ToolCallID: "call_test_123",
			Content:    []ContentPart{{Type: ContentTypeText, Text: "file content 2"}},
		},
	}

	contents, _ := p.buildGeminiContents(messages, false)
	if len(contents) != 3 {
		t.Fatalf("Expected 3 turns (user, model, merged tool response turn), got %d", len(contents))
	}

	// Model turn should have FunctionCall with thoughtSignature restored
	if contents[1].Parts[0].ThoughtSignature != fakeSig {
		t.Errorf("Expected thoughtSignature %q, got %q", fakeSig, contents[1].Parts[0].ThoughtSignature)
	}

	// The two tool responses should be merged into turn 2
	if len(contents[2].Parts) != 2 {
		t.Errorf("Expected 2 merged tool responses in turn 2, got %d", len(contents[2].Parts))
	}
}

func TestProcessTextStreamExtraction(t *testing.T) {
	p := NewGoogleProvider([]string{"dummy"})
	tools := []Tool{
		{
			Type: "function",
			Functions: []Tool{
				{
					Name: "read_file",
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"filePath":  map[string]any{"type": "string"},
							"startLine": map[string]any{"type": "integer"},
							"endLine":   map[string]any{"type": "integer"},
						},
						"required": []any{"filePath"},
					},
				},
			},
		},
	}

	// 1. Test standardized <tool_calls> tag with JSON array
	t.Run("Standardized tool_calls array", func(t *testing.T) {
		var extractedCalls []ToolCall
		var yieldedContent string

		raw := "I will inspect the file.\n<tool_calls>\n[\n  {\n    \"name\": \"read_file\",\n    \"arguments\": {\"filePath\": \"README.md\", \"startLine\": 1, \"endLine\": 50}\n  }\n]\n</tool_calls>\nDone."
		pending := p.processTextStream("", raw, tools, func(tc ToolCall) {
			extractedCalls = append(extractedCalls, tc)
		}, func(content string) {
			yieldedContent += content
		})
		p.flushTextStream(pending, tools, func(tc ToolCall) {
			extractedCalls = append(extractedCalls, tc)
		}, func(content string) {
			yieldedContent += content
		})

		if len(extractedCalls) != 1 {
			t.Fatalf("Expected 1 extracted call, got %d", len(extractedCalls))
		}
		if extractedCalls[0].Function.Name != "read_file" {
			t.Errorf("Expected func name read_file, got %s", extractedCalls[0].Function.Name)
		}
		if !strings.Contains(yieldedContent, "I will inspect the file.") || !strings.Contains(yieldedContent, "Done.") {
			t.Errorf("Unexpected yielded content: %q", yieldedContent)
		}
		if strings.Contains(yieldedContent, "<tool_calls>") {
			t.Errorf("Tag <tool_calls> should not be in yielded content: %q", yieldedContent)
		}
	})

	// 2. Test legacy [Called Tool: ...] format
	t.Run("Legacy Called Tool format", func(t *testing.T) {
		var extractedCalls []ToolCall
		var yieldedContent string

		raw := `[Called Tool: read_file with arguments: {"endLine":100,"filePath":"/home/neuxbane/Projects/github.com/Ollama-One/README.md","startLine":1}]`
		pending := p.processTextStream("", raw, tools, func(tc ToolCall) {
			extractedCalls = append(extractedCalls, tc)
		}, func(content string) {
			yieldedContent += content
		})
		p.flushTextStream(pending, tools, func(tc ToolCall) {
			extractedCalls = append(extractedCalls, tc)
		}, func(content string) {
			yieldedContent += content
		})

		if len(extractedCalls) != 1 {
			t.Fatalf("Expected 1 extracted call, got %d", len(extractedCalls))
		}
		if extractedCalls[0].Function.Name != "read_file" {
			t.Errorf("Expected func name read_file, got %s", extractedCalls[0].Function.Name)
		}
		if strings.Contains(yieldedContent, "[Called Tool:") {
			t.Errorf("Legacy tag should not be in yielded content: %q", yieldedContent)
		}
	})

	// 3. Test chunked streaming across multiple chunks
	t.Run("Chunked streaming", func(t *testing.T) {
		var extractedCalls []ToolCall
		var yieldedContent string

		chunks := []string{
			"Thinking...\n<tool",
			"_calls>[\n",
			`{"name": "read_file", "arguments": {"filePath": "main.go"}}`,
			"\n]</tool_calls>",
		}

		pending := ""
		for _, chunk := range chunks {
			pending = p.processTextStream(pending, chunk, tools, func(tc ToolCall) {
				extractedCalls = append(extractedCalls, tc)
			}, func(content string) {
				yieldedContent += content
			})
		}
		p.flushTextStream(pending, tools, func(tc ToolCall) {
			extractedCalls = append(extractedCalls, tc)
		}, func(content string) {
			yieldedContent += content
		})

		if len(extractedCalls) != 1 {
			t.Fatalf("Expected 1 extracted call from chunks, got %d", len(extractedCalls))
		}
		if extractedCalls[0].Function.Name != "read_file" {
			t.Errorf("Expected read_file, got %s", extractedCalls[0].Function.Name)
		}
		if !strings.Contains(yieldedContent, "Thinking...") {
			t.Errorf("Expected Thinking... in content, got %q", yieldedContent)
		}
	})
}

func TestStrictModelRouting(t *testing.T) {
	wsModel := "google/ws/gemini-3.8-live-extended-thinking"
	if !strings.HasPrefix(wsModel, "google/ws/") {
		t.Errorf("Expected %q to have google/ws/ prefix", wsModel)
	}

	restModel := "google/gemini-2.0-flash"
	if strings.HasPrefix(restModel, "google/ws/") {
		t.Errorf("Did not expect %q to have google/ws/ prefix", restModel)
	}
}

func TestLiveConnSessionMatching(t *testing.T) {
	liveConnsLock.Lock()
	liveConns = make(map[string]*liveConnSession)
	liveConnsLock.Unlock()

	// 1. Expired session is cleaned up
	expiredSess := &liveConnSession{
		expiresAt: time.Now().Add(-1 * time.Minute),
	}
	liveConnsLock.Lock()
	liveConns["call_expired"] = expiredSess
	liveConnsLock.Unlock()

	cleanExpiredLiveConns()

	liveConnsLock.Lock()
	if _, exists := liveConns["call_expired"]; exists {
		t.Errorf("Expected call_expired to be cleaned up")
	}
	liveConnsLock.Unlock()

	// 2. Active session is retained
	activeSess := &liveConnSession{
		expiresAt: time.Now().Add(2 * time.Minute),
	}
	liveConnsLock.Lock()
	liveConns["call_active_1"] = activeSess
	liveConns["call_active_2"] = activeSess
	liveConnsLock.Unlock()

	cleanExpiredLiveConns()

	liveConnsLock.Lock()
	if _, exists := liveConns["call_active_1"]; !exists {
		t.Errorf("Expected call_active_1 to remain active")
	}
	if _, exists := liveConns["call_active_2"]; !exists {
		t.Errorf("Expected call_active_2 to remain active")
	}
	liveConnsLock.Unlock()
}

func TestIsRetryableError(t *testing.T) {
	p := NewGoogleProvider([]string{"key1"})

	if !p.isRetryableError(fmt.Errorf("gemini api error (status 429): Resource exhausted")) {
		t.Errorf("Expected 429 to be retryable")
	}
	if !p.isRetryableError(fmt.Errorf("gemini api error (status 503): Service Unavailable")) {
		t.Errorf("Expected 503 to be retryable")
	}
	if !p.isRetryableError(fmt.Errorf("empty response received from Gemini (0 content, 0 tool calls)")) {
		t.Errorf("Expected empty response to be retryable")
	}
	if !p.isRetryableError(fmt.Errorf("unexpected EOF")) {
		t.Errorf("Expected EOF to be retryable")
	}
	if !p.isRetryableError(fmt.Errorf("read: connection reset by peer")) {
		t.Errorf("Expected connection reset to be retryable")
	}
	if p.isRetryableError(fmt.Errorf("gemini api error (status 400): Invalid JSON")) {
		t.Errorf("Did not expect 400 to be retryable")
	}

	// Single key: 401/403 should not be retryable
	if p.isRetryableError(fmt.Errorf("gemini api error (status 401): Unauthorized")) {
		t.Errorf("Did not expect 401 to be retryable with single key")
	}

	// Multiple keys: 401/403 should be retryable (key rotation)
	pMulti := NewGoogleProvider([]string{"key1", "key2"})
	if !pMulti.isRetryableError(fmt.Errorf("gemini api error (status 401): Unauthorized")) {
		t.Errorf("Expected 401 to be retryable with multiple keys")
	}
}

func TestGetKeyForAttempt(t *testing.T) {
	p := NewGoogleProvider([]string{"pool_key1", "pool_key2"})

	// Attempt 0 with explicit key
	if k := p.getKeyForAttempt(0, "explicit_key"); k != "explicit_key" {
		t.Errorf("Expected explicit_key, got %s", k)
	}

	// Attempt 1 with multiple pool keys rotates
	k1 := p.getKeyForAttempt(1, "explicit_key")
	if k1 != "pool_key1" && k1 != "pool_key2" {
		t.Errorf("Expected pool key rotation on retry, got %s", k1)
	}

	// Comma separated keys in reqKey
	multiReqKey := "req_keyA,req_keyB"
	if k := p.getKeyForAttempt(0, multiReqKey); k != "req_keyA" {
		t.Errorf("Expected req_keyA, got %s", k)
	}
	if k := p.getKeyForAttempt(1, multiReqKey); k != "req_keyB" {
		t.Errorf("Expected req_keyB, got %s", k)
	}
}

func TestAutoRetryOnRateLimit(t *testing.T) {
	var requestCount int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&requestCount, 1)
		if count == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte(`{"error":{"code":429,"message":"Resource exhausted"}}`))
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Success after rate limit retry\"}]}}]}\n\n")
	}))
	defer server.Close()

	p := NewGoogleProvider([]string{"key1", "key2"})
	p.BaseURL = server.URL

	resp, err := p.Chat(context.Background(), &CompletionRequest{
		Model: "google/gemini-2.5-flash",
	}, nil)

	if err != nil {
		t.Fatalf("Expected Chat to succeed after retry, got error: %v", err)
	}
	if atomic.LoadInt32(&requestCount) != 2 {
		t.Errorf("Expected 2 requests made, got %d", atomic.LoadInt32(&requestCount))
	}
	if !strings.Contains(resp.Content, "Success after rate limit retry") {
		t.Errorf("Unexpected content: %s", resp.Content)
	}
}

func TestAutoRetryOnEmptyResponse(t *testing.T) {
	var requestCount int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&requestCount, 1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if count == 1 {
			// First response is completely empty candidates
			fmt.Fprintf(w, "data: {\"candidates\":[]}\n\n")
			return
		}
		// Second response has content
		fmt.Fprintf(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Recovered from empty turn\"}]}}]}\n\n")
	}))
	defer server.Close()

	p := NewGoogleProvider([]string{"test-key"})
	p.BaseURL = server.URL

	resp, err := p.Chat(context.Background(), &CompletionRequest{
		Model: "google/gemini-2.5-flash",
	}, nil)

	if err != nil {
		t.Fatalf("Expected Chat to succeed after retry, got error: %v", err)
	}
	if atomic.LoadInt32(&requestCount) != 2 {
		t.Errorf("Expected 2 requests made, got %d", atomic.LoadInt32(&requestCount))
	}
	if !strings.Contains(resp.Content, "Recovered from empty turn") {
		t.Errorf("Unexpected content: %s", resp.Content)
	}
}



