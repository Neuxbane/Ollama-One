package providers

import (
	"context"
	"strings"
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

	// Live mode: turns are formatted as text to conform to Google Live WebSocket schema and avoid 1007
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
	if liveContents[1].Role != "model" || !strings.Contains(liveContents[1].Parts[0].Text, "[Called Tool: read_file") {
		t.Errorf("Turn 1 mismatch: %+v", liveContents[1])
	}
	if liveContents[2].Role != "user" || !strings.Contains(liveContents[2].Parts[0].Text, "[Tool Result for read_file]:\npackage main") {
		t.Errorf("Turn 2 mismatch: %+v", liveContents[2])
	}

	// REST mode: tool role becomes function with structured functionResponse part
	restContents, _ := p.buildGeminiContents(messages, false)
	if len(restContents) != 3 {
		t.Fatalf("Expected 3 turns, got %d", len(restContents))
	}
	if restContents[1].Role != "model" || restContents[1].Parts[0].FunctionCall == nil || restContents[1].Parts[0].FunctionCall.Name != "read_file" {
		t.Errorf("Turn 1 REST mismatch: %+v", restContents[1])
	}
	if restContents[2].Role != "function" || restContents[2].Parts[0].FunctionResponse == nil {
		t.Errorf("Turn 2 REST mismatch: %+v", restContents[2])
	}
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

func TestTruncateMiddle(t *testing.T) {
	shortStr := "Hello world"
	if res := truncateMiddle(shortStr, 100); res != shortStr {
		t.Errorf("Expected short string unchanged, got %q", res)
	}

	// 50,000 char string
	longStr := strings.Repeat("A", 25000) + strings.Repeat("Z", 25000)
	truncated := truncateMiddle(longStr, 16000)

	if len(truncated) > 16200 {
		t.Errorf("Expected truncated string length <= 16200, got %d", len(truncated))
	}
	if !strings.HasPrefix(truncated, "AAAA") {
		t.Errorf("Expected start of string preserved")
	}
	if !strings.HasSuffix(truncated, "ZZZZ") {
		t.Errorf("Expected end of string preserved")
	}
	if !strings.Contains(truncated, "[middle truncated:") {
		t.Errorf("Expected middle truncation marker, got %q", truncated)
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


