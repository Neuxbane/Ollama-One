package middleware

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/Neuxbane/Ollama-One/providers"
)

func generateDummyTools(count int) []providers.Tool {
	var tools []providers.Tool
	for i := 1; i <= count; i++ {
		tools = append(tools, providers.Tool{
			Type:        "function",
			Name:        fmt.Sprintf("tool_%d", i),
			Description: fmt.Sprintf("Description for tool number %d", i),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"param1": map[string]any{"type": "string", "description": "dummy parameter"},
				},
			},
		})
	}
	return tools
}

func TestWarpToolsThresholdBypass(t *testing.T) {
	// If tools count <= top_k (e.g. 5 <= 10), warp_tools must NOT engage
	tools := generateDummyTools(5)
	req := &providers.CompletionRequest{
		Model: "gemini-2.5-pro",
		Tools: tools,
	}

	err := WarpToolsModifier(req, []string{"top_k=10"})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(req.Tools) != 5 {
		t.Errorf("Expected tools to remain unchanged (5 tools), got %d", len(req.Tools))
	}
	if req.Metadata != nil && req.Metadata["warp_tools"] != nil {
		t.Errorf("Expected warp_tools to not be configured in metadata")
	}

	// Exact threshold check: 10 <= 10 should also bypass
	tools10 := generateDummyTools(10)
	req10 := &providers.CompletionRequest{
		Model: "gemini-2.5-pro",
		Tools: tools10,
	}
	err = WarpToolsModifier(req10, []string{"top_k=10"})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if len(req10.Tools) != 10 {
		t.Errorf("Expected 10 tools to remain unchanged when tools == top_k, got %d", len(req10.Tools))
	}
}

func TestWarpToolsActivationAboveThreshold(t *testing.T) {
	// If tools count > top_k (e.g. 15 > 10), warp_tools MUST engage
	tools := generateDummyTools(15)
	req := &providers.CompletionRequest{
		Model: "gemini-2.5-pro",
		Tools: tools,
	}

	err := WarpToolsModifier(req, []string{"top_k=10"})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(req.Tools) != 2 {
		t.Fatalf("Expected tools to be replaced with 2 meta-tools (searchTools, execute), got %d", len(req.Tools))
	}

	toolNames := map[string]bool{req.Tools[0].Name: true, req.Tools[1].Name: true}
	if !toolNames["searchTools"] || !toolNames["execute"] {
		t.Errorf("Expected meta-tools searchTools and execute, got: %+v", req.Tools)
	}

	cfg, ok := GetWarpToolsConfig(req)
	if !ok || cfg == nil {
		t.Fatalf("Expected WarpToolsConfig in metadata")
	}
	if len(cfg.OriginalTools) != 15 {
		t.Errorf("Expected 15 original tools in config, got %d", len(cfg.OriginalTools))
	}
	if cfg.TopK != 10 {
		t.Errorf("Expected TopK to be 10, got %d", cfg.TopK)
	}
}

func TestSemanticToolIndexSearch(t *testing.T) {
	tools := []providers.Tool{
		{
			Type:        "function",
			Name:        "get_current_weather",
			Description: "Get the current temperature, humidity, and weather conditions for a location",
			Parameters:  map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
		},
		{
			Type:        "function",
			Name:        "read_local_file",
			Description: "Read the entire contents of a file from disk given its file path",
			Parameters:  map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}},
		},
		{
			Type:        "function",
			Name:        "execute_database_sql",
			Description: "Run a read-only SQL query against the customer database and return rows",
			Parameters:  map[string]any{"type": "object", "properties": map[string]any{"query": map[string]any{"type": "string"}}},
		},
		{
			Type:        "function",
			Name:        "send_email_notification",
			Description: "Send an email alert to a user with subject and body",
			Parameters:  map[string]any{"type": "object", "properties": map[string]any{"to": map[string]any{"type": "string"}}},
		},
	}

	index := NewSemanticToolIndex(tools)

	// Query 1: Weather
	resultsWeather := index.Search("what is the weather temperature outside", 2)
	if len(resultsWeather) == 0 || resultsWeather[0].Name != "get_current_weather" {
		t.Errorf("Expected get_current_weather for weather query, got: %+v", resultsWeather)
	}

	// Query 2: File reading
	resultsFile := index.Search("inspect and read file from filesystem", 2)
	if len(resultsFile) == 0 || resultsFile[0].Name != "read_local_file" {
		t.Errorf("Expected read_local_file for file query, got: %+v", resultsFile)
	}

	// Query 3: Database query
	resultsDB := index.Search("lookup records in table with SQL", 2)
	if len(resultsDB) == 0 || resultsDB[0].Name != "execute_database_sql" {
		t.Errorf("Expected execute_database_sql for DB query, got: %+v", resultsDB)
	}
}

// MockChatProvider simulates multi-turn model interaction for warp_tools
type MockWarpChatProvider struct {
	turn int
}

func (m *MockWarpChatProvider) ListModels(ctx context.Context) ([]providers.ModelInfo, error) {
	return nil, nil
}
func (m *MockWarpChatProvider) ListModelsWithKey(ctx context.Context, apiKey string) ([]providers.ModelInfo, error) {
	return nil, nil
}

func (m *MockWarpChatProvider) Chat(ctx context.Context, req *providers.CompletionRequest, onChunk func(*providers.CompletionResponse)) (*providers.CompletionResponse, error) {
	m.turn++

	// Turn 1: Model sees only searchTools and execute, so it calls searchTools
	if m.turn == 1 {
		return &providers.CompletionResponse{
			ToolCalls: []providers.ToolCall{
				{
					ID:   "call_search_1",
					Type: "function",
					Function: providers.FunctionCall{
						Name:      "searchTools",
						Arguments: `{"query": "weather forecast"}`,
					},
				},
			},
		}, nil
	}

	// Turn 2: Model received tool schemas from proxy, now calls execute
	return &providers.CompletionResponse{
		ToolCalls: []providers.ToolCall{
			{
				ID:   "call_exec_1",
				Type: "function",
				Function: providers.FunctionCall{
					Name:      "execute",
					Arguments: `{"tools": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]}`,
				},
			},
		},
	}, nil
}

func TestExecuteWarpToolsLoop(t *testing.T) {
	tools := []providers.Tool{
		{
			Type:        "function",
			Name:        "get_weather",
			Description: "Get the current weather forecast for a city",
		},
	}
	for i := 2; i <= 20; i++ {
		tools = append(tools, providers.Tool{
			Type:        "function",
			Name:        fmt.Sprintf("other_tool_%d", i),
			Description: fmt.Sprintf("Other unrelated tool %d", i),
		})
	}

	req := &providers.CompletionRequest{
		Model: "gemini-2.5-pro",
		Tools: tools,
		Messages: []providers.Message{
			{Role: "user", Content: []providers.ContentPart{{Type: providers.ContentTypeText, Text: "Weather in Tokyo?"}}},
		},
	}

	// Apply modifier with top_k=5
	err := WarpToolsModifier(req, []string{"top_k=5"})
	if err != nil {
		t.Fatalf("WarpToolsModifier error: %v", err)
	}

	mock := &MockWarpChatProvider{}
	resp, err := ExecuteChat(context.Background(), mock, req, nil)
	if err != nil {
		t.Fatalf("ExecuteChat error: %v", err)
	}

	// Verify that execute was unwrapped to native get_weather tool call
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("Expected 1 unwrapped tool call, got %d", len(resp.ToolCalls))
	}

	tc := resp.ToolCalls[0]
	if tc.Function.Name != "get_weather" {
		t.Errorf("Expected unwrapped tool name 'get_weather', got %q", tc.Function.Name)
	}
	if !strings.Contains(tc.Function.Arguments, "Tokyo") {
		t.Errorf("Expected Tokyo in arguments, got %s", tc.Function.Arguments)
	}
}
