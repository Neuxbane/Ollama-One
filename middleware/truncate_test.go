package middleware

import (
	"strings"
	"testing"

	"github.com/Neuxbane/Ollama-One/providers"
)

func TestParseModelDirectives(t *testing.T) {
	tests := []struct {
		name           string
		rawModel       string
		expectedBase   string
		expectedCount  int
		expectedName   string
		expectedArgs   []string
	}{
		{
			name:          "Full model with truncate all 4000",
			rawModel:      "google/ws/gemini-3.1-flash-live-preview:truncate(all, 4000)",
			expectedBase:  "google/ws/gemini-3.1-flash-live-preview",
			expectedCount: 1,
			expectedName:  "truncate",
			expectedArgs:  []string{"all", "4000"},
		},
		{
			name:          "Model with single arg truncate",
			rawModel:      "gemini-2.5-flash:truncate(2000)",
			expectedBase:  "gemini-2.5-flash",
			expectedCount: 1,
			expectedName:  "truncate",
			expectedArgs:  []string{"2000"},
		},
		{
			name:          "Model with quotes and spaces",
			rawModel:      "google/gemini-2.5-flash:truncate('tools', 1000)",
			expectedBase:  "google/gemini-2.5-flash",
			expectedCount: 1,
			expectedName:  "truncate",
			expectedArgs:  []string{"tools", "1000"},
		},
		{
			name:          "Plain model with no directives",
			rawModel:      "google/gemini-2.5-flash",
			expectedBase:  "google/gemini-2.5-flash",
			expectedCount: 0,
		},
		{
			name:          "Model with latest tag and no directives",
			rawModel:      "gemini-2.0-flash:latest",
			expectedBase:  "gemini-2.0-flash:latest",
			expectedCount: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			base, directives := ParseModelDirectives(tc.rawModel)
			if base != tc.expectedBase {
				t.Errorf("Expected base model %q, got %q", tc.expectedBase, base)
			}
			if len(directives) != tc.expectedCount {
				t.Fatalf("Expected %d directives, got %d", tc.expectedCount, len(directives))
			}
			if tc.expectedCount > 0 {
				if directives[0].Name != tc.expectedName {
					t.Errorf("Expected directive name %q, got %q", tc.expectedName, directives[0].Name)
				}
				if len(directives[0].Args) != len(tc.expectedArgs) {
					t.Fatalf("Expected %d args, got %d: %v", len(tc.expectedArgs), len(directives[0].Args), directives[0].Args)
				}
				for i, arg := range tc.expectedArgs {
					if directives[0].Args[i] != arg {
						t.Errorf("Arg %d: expected %q, got %q", i, arg, directives[0].Args[i])
					}
				}
			}
		})
	}
}

func TestTruncateText(t *testing.T) {
	// 1. Shorter than limit
	short := "Hello world"
	if res := TruncateText(short, 20); res != short {
		t.Errorf("Expected unchanged, got %q", res)
	}

	// 2. Longer than limit
	long := strings.Repeat("A", 100)
	res := TruncateText(long, 50)
	if len([]rune(res)) > 50 {
		t.Errorf("Expected length <= 50, got %d", len([]rune(res)))
	}
	if !strings.Contains(res, "[truncated") {
		t.Errorf("Expected truncation marker in result: %q", res)
	}

	// 3. UTF-8 runes test
	emojis := strings.Repeat("🌟", 50)
	resEmoji := TruncateText(emojis, 20)
	if len([]rune(resEmoji)) > 20 {
		t.Errorf("Expected rune count <= 20, got %d", len([]rune(resEmoji)))
	}
}

func TestTruncateModifierAll(t *testing.T) {
	req := &providers.CompletionRequest{
		Model:             "google/ws/gemini-3.1-flash-live-preview:truncate(all, 100)",
		SystemInstruction: strings.Repeat("S", 200),
		Messages: []providers.Message{
			{
				Role: "user",
				Content: []providers.ContentPart{
					{Type: providers.ContentTypeText, Text: strings.Repeat("U", 300)},
				},
			},
			{
				Role:       "tool",
				Name:       "read_file",
				ToolCallID: "call_123",
				Content: []providers.ContentPart{
					{Type: providers.ContentTypeText, Text: strings.Repeat("F", 5000)}, // Function response
				},
			},
		},
	}

	err := Process(req)
	if err != nil {
		t.Fatalf("Unexpected Process error: %v", err)
	}

	// Model should be cleaned
	if req.Model != "google/ws/gemini-3.1-flash-live-preview" {
		t.Errorf("Expected clean model, got %q", req.Model)
	}

	// System instruction should be <= 100 runes
	if len([]rune(req.SystemInstruction)) > 100 {
		t.Errorf("SystemInstruction exceeded limit: %d", len([]rune(req.SystemInstruction)))
	}

	// User input should be <= 100 runes
	userText := req.Messages[0].Content[0].Text
	if len([]rune(userText)) > 100 {
		t.Errorf("User message exceeded limit: %d", len([]rune(userText)))
	}

	// Function response should be <= 100 runes
	funcText := req.Messages[1].Content[0].Text
	if len([]rune(funcText)) > 100 {
		t.Errorf("Function response exceeded limit: %d", len([]rune(funcText)))
	}
	if !strings.Contains(funcText, "[truncated") {
		t.Errorf("Expected truncation indicator in function response")
	}
}

func TestTruncateModifierToolsOnly(t *testing.T) {
	req := &providers.CompletionRequest{
		Model: "google/gemini-2.5-flash:truncate(tools, 50)",
		Messages: []providers.Message{
			{
				Role: "user",
				Content: []providers.ContentPart{
					{Type: providers.ContentTypeText, Text: strings.Repeat("U", 200)}, // Should NOT be truncated
				},
			},
			{
				Role:       "tool",
				Name:       "grep",
				ToolCallID: "call_abc",
				Content: []providers.ContentPart{
					{Type: providers.ContentTypeText, Text: strings.Repeat("T", 300)}, // Should be truncated
				},
			},
		},
	}

	err := Process(req)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// User message should remain untruncated (200 runes)
	userText := req.Messages[0].Content[0].Text
	if len([]rune(userText)) != 200 {
		t.Errorf("Expected user message to remain 200 chars, got %d", len([]rune(userText)))
	}

	// Tool message should be truncated to <= 50 runes
	toolText := req.Messages[1].Content[0].Text
	if len([]rune(toolText)) > 50 {
		t.Errorf("Expected tool response <= 50 chars, got %d", len([]rune(toolText)))
	}
}
