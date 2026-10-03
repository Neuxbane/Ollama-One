package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestDeepSeekListModelsFallback(t *testing.T) {
	p := NewDeepSeekProvider(nil)
	models, err := p.ListModels(context.Background())
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if len(models) == 0 {
		t.Fatalf("Expected fallback models, got 0")
	}

	foundChat := false
	foundReasoner := false
	for _, m := range models {
		if strings.Contains(m.ID, "deepseek-chat") {
			foundChat = true
		}
		if strings.Contains(m.ID, "deepseek-reasoner") {
			foundReasoner = true
		}
	}
	if !foundChat || !foundReasoner {
		t.Errorf("Expected deepseek-chat and deepseek-reasoner in fallback models")
	}
}

func TestDeepSeekListModelsFromAPI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer test-api-key" {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"id": "deepseek-chat", "object": "model", "owned_by": "deepseek"},
				{"id": "deepseek-reasoner", "object": "model", "owned_by": "deepseek"},
			},
		})
	}))
	defer server.Close()

	p := NewDeepSeekProvider([]string{"test-api-key"})
	p.BaseURL = server.URL

	models, err := p.ListModelsWithKey(context.Background(), "test-api-key")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	modelIDs := make(map[string]bool)
	for _, m := range models {
		modelIDs[m.ID] = true
	}

	if !modelIDs["deepseek-chat"] || !modelIDs["deepseek/deepseek-chat"] {
		t.Errorf("Expected deepseek-chat and deepseek/deepseek-chat in models list: %+v", modelIDs)
	}
	if !modelIDs["deepseek-reasoner"] || !modelIDs["deepseek/deepseek-reasoner"] {
		t.Errorf("Expected deepseek-reasoner and deepseek/deepseek-reasoner in models list: %+v", modelIDs)
	}
}

func TestDeepSeekChatNonStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("Authorization") != "Bearer secret-key" {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		var reqBody map[string]any
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if reqBody["model"] != "deepseek-chat" {
			t.Errorf("Expected target model deepseek-chat, got %v", reqBody["model"])
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-123",
			"object":  "chat.completion",
			"created": 1700000000,
			"model":   "deepseek-chat",
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": "Hello from DeepSeek!",
					},
					"finish_reason": "stop",
				},
			},
		})
	}))
	defer server.Close()

	p := NewDeepSeekProvider(nil)
	p.BaseURL = server.URL

	req := &CompletionRequest{
		APIKey: "secret-key",
		Model:  "deepseek/deepseek-chat",
		Messages: []Message{
			{
				Role:    "user",
				Content: []ContentPart{{Type: ContentTypeText, Text: "Hi"}},
			},
		},
	}

	resp, err := p.Chat(context.Background(), req, nil)
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if resp.Content != "Hello from DeepSeek!" {
		t.Errorf("Expected 'Hello from DeepSeek!', got %q", resp.Content)
	}
}

func TestDeepSeekChatStreamingAndReasoning(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			http.NotFound(w, r)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatalf("ResponseWriter is not a Flusher")
		}

		// Reasoning delta chunk
		chunk1 := `{"id":"1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Thinking deeply..."}}]}`
		fmt.Fprintf(w, "data: %s\n\n", chunk1)
		flusher.Flush()

		// Content delta chunk
		chunk2 := `{"id":"2","choices":[{"index":0,"delta":{"content":"Answer: 42"}}]}`
		fmt.Fprintf(w, "data: %s\n\n", chunk2)
		flusher.Flush()

		// Done chunk
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	p := NewDeepSeekProvider(nil)
	p.BaseURL = server.URL

	req := &CompletionRequest{
		APIKey: "sk-test",
		Model:  "deepseek-reasoner",
		Stream: true,
		Messages: []Message{
			{
				Role:    "user",
				Content: []ContentPart{{Type: ContentTypeText, Text: "Solve this"}},
			},
		},
	}

	var streamedThoughts strings.Builder
	var streamedContent strings.Builder

	resp, err := p.Chat(context.Background(), req, func(c *CompletionResponse) {
		if c.Thought != "" {
			streamedThoughts.WriteString(c.Thought)
		}
		if c.Content != "" {
			streamedContent.WriteString(c.Content)
		}
	})

	if err != nil {
		t.Fatalf("Streaming chat error: %v", err)
	}

	if streamedThoughts.String() != "Thinking deeply..." {
		t.Errorf("Expected streamed thought 'Thinking deeply...', got %q", streamedThoughts.String())
	}
	if streamedContent.String() != "Answer: 42" {
		t.Errorf("Expected streamed content 'Answer: 42', got %q", streamedContent.String())
	}
	if resp.Thought != "Thinking deeply..." {
		t.Errorf("Expected final thought 'Thinking deeply...', got %q", resp.Thought)
	}
	if resp.Content != "Answer: 42" {
		t.Errorf("Expected final content 'Answer: 42', got %q", resp.Content)
	}
}

func TestDeepSeekToolCallingStreaming(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			http.NotFound(w, r)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)

		// Tool call start
		chunk1 := `{"id":"1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}`
		fmt.Fprintf(w, "data: %s\n\n", chunk1)
		flusher.Flush()

		// Tool call args part 1
		chunk2 := `{"id":"2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc\":"}}]}}]}`
		fmt.Fprintf(w, "data: %s\n\n", chunk2)
		flusher.Flush()

		// Tool call args part 2
		chunk3 := `{"id":"3","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Tokyo\"}"}}]}}]}`
		fmt.Fprintf(w, "data: %s\n\n", chunk3)
		flusher.Flush()

		// Finish
		chunk4 := `{"id":"4","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`
		fmt.Fprintf(w, "data: %s\n\n", chunk4)
		flusher.Flush()

		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	p := NewDeepSeekProvider(nil)
	p.BaseURL = server.URL

	req := &CompletionRequest{
		APIKey: "sk-test",
		Model:  "deepseek-chat",
		Stream: true,
		Tools: []Tool{
			{
				Type:        "function",
				Name:        "get_weather",
				Description: "Get weather in location",
			},
		},
		Messages: []Message{
			{Role: "user", Content: []ContentPart{{Type: ContentTypeText, Text: "Weather in Tokyo?"}}},
		},
	}

	var yieldedToolCalls []ToolCall
	resp, err := p.Chat(context.Background(), req, func(c *CompletionResponse) {
		if len(c.ToolCalls) > 0 {
			yieldedToolCalls = append(yieldedToolCalls, c.ToolCalls...)
		}
	})

	if err != nil {
		t.Fatalf("Tool call stream error: %v", err)
	}

	if len(resp.ToolCalls) != 1 {
		t.Fatalf("Expected 1 tool call in resp, got %d", len(resp.ToolCalls))
	}
	if len(yieldedToolCalls) != 1 {
		t.Fatalf("Expected 1 yielded tool call, got %d", len(yieldedToolCalls))
	}

	tc := resp.ToolCalls[0]
	if tc.ID != "call_abc" || tc.Function.Name != "get_weather" || tc.Function.Arguments != `{"loc":"Tokyo"}` {
		t.Errorf("Unexpected tool call contents: %+v", tc)
	}
}

func TestDeepSeekRetryOn429(t *testing.T) {
	var attempts int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		att := atomic.AddInt32(&attempts, 1)
		if att == 1 {
			http.Error(w, `{"error":{"message":"Rate limit reached"}}`, http.StatusTooManyRequests)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-retry",
			"object":  "chat.completion",
			"created": 1700000000,
			"model":   "deepseek-chat",
			"choices": []map[string]any{
				{
					"index":   0,
					"message": map[string]any{"role": "assistant", "content": "Success after retry"},
				},
			},
		})
	}))
	defer server.Close()

	p := NewDeepSeekProvider(nil)
	p.BaseURL = server.URL

	req := &CompletionRequest{
		APIKey:   "sk-test",
		Model:    "deepseek-chat",
		Messages: []Message{{Role: "user", Content: []ContentPart{{Type: ContentTypeText, Text: "Hello"}}}},
	}

	resp, err := p.Chat(context.Background(), req, nil)
	if err != nil {
		t.Fatalf("Expected retry to succeed, got error: %v", err)
	}
	if resp.Content != "Success after retry" {
		t.Errorf("Expected 'Success after retry', got %q", resp.Content)
	}
	if atomic.LoadInt32(&attempts) != 2 {
		t.Errorf("Expected 2 attempts, got %d", atomic.LoadInt32(&attempts))
	}
}
