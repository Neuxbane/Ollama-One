package providers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestGeminiListModelsPagination(t *testing.T) {
	page1Requested := false
	page2Requested := false

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1beta/models" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			http.Error(w, "not found", http.StatusNotFound)
			return
		}

		key := r.URL.Query().Get("key")
		if key != "test-api-key" {
			t.Errorf("unexpected key: %s", key)
		}

		pageToken := r.URL.Query().Get("pageToken")
		w.Header().Set("Content-Type", "application/json")

		if pageToken == "" {
			page1Requested = true
			resp := map[string]any{
				"models": []map[string]any{
					{
						"name":                       "models/gemini-2.0-flash",
						"displayName":                "Gemini 2.0 Flash",
						"supportedGenerationMethods": []string{"generateContent"},
						"thinking":                   false,
					},
					{
						"name":                       "models/gemini-2.5-pro",
						"displayName":                "Gemini 2.5 Pro",
						"supportedGenerationMethods": []string{"generateContent"},
						"thinking":                   true,
					},
					{
						"name":                       "models/embedding-001",
						"displayName":                "Embedding 001",
						"supportedGenerationMethods": []string{"embedContent"},
					},
				},
				"nextPageToken": "token-page-2",
			}
			json.NewEncoder(w).Encode(resp)
		} else if pageToken == "token-page-2" {
			page2Requested = true
			resp := map[string]any{
				"models": []map[string]any{
					{
						"name":                       "models/gemini-3.0-ultra",
						"displayName":                "Gemini 3.0 Ultra",
						"supportedGenerationMethods": []string{"generateContent"},
						"thinking":                   true,
					},
				},
				"nextPageToken": "",
			}
			json.NewEncoder(w).Encode(resp)
		} else {
			t.Errorf("unexpected pageToken: %s", pageToken)
			http.Error(w, "bad token", http.StatusBadRequest)
		}
	}))
	defer server.Close()

	provider := &GeminiProvider{
		BaseProvider: BaseProvider{APIKeys: []string{"test-api-key"}},
		BaseURL:      server.URL,
	}

	models, err := provider.ListModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !page1Requested {
		t.Errorf("expected page 1 to be requested")
	}
	if !page2Requested {
		t.Errorf("expected page 2 to be requested")
	}

	if len(models) != 4 {
		t.Fatalf("expected 4 models without method filter, got %d", len(models))
	}

	expectedIDs := []string{"gemini-2.0-flash", "gemini-2.5-pro", "embedding-001", "gemini-3.0-ultra"}
	for i, id := range expectedIDs {
		if models[i].ID != id {
			t.Errorf("expected model %d ID to be %s, got %s", i, id, models[i].ID)
		}
	}
}

func TestSanitizeGeminiSchema(t *testing.T) {
	input := map[string]any{
		"type":     "object",
		"$comment": "Top level comment",
		"title":    "My Tool",
		"properties": map[string]any{
			"filePath": map[string]any{
				"type":             "string",
				"description":      "Path to file",
				"$comment":         "Nested comment",
				"enumDescriptions": []any{"desc 1"},
			},
		},
	}

	sanitized := sanitizeGeminiSchemaMap(input)

	if _, exists := sanitized["$comment"]; exists {
		t.Errorf("expected $comment to be stripped from top level")
	}
	if _, exists := sanitized["title"]; exists {
		t.Errorf("expected title to be stripped from top level")
	}

	props, ok := sanitized["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected properties map to exist")
	}

	fp, ok := props["filePath"].(map[string]any)
	if !ok {
		t.Fatalf("expected filePath map to exist")
	}

	if _, exists := fp["$comment"]; exists {
		t.Errorf("expected $comment to be stripped from nested property")
	}
	if _, exists := fp["enumDescriptions"]; exists {
		t.Errorf("expected enumDescriptions to be stripped from nested property")
	}
	if fp["type"] != "string" {
		t.Errorf("expected type to be preserved, got %v", fp["type"])
	}

	// Test required filtering when properties are missing undefined keys
	invalidReqInput := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"id": map[string]any{"type": "string"},
		},
		"required": []any{"id", "missingKey"},
	}

	sanitizedReq := sanitizeGeminiSchemaMap(invalidReqInput)
	reqSlice, ok := sanitizedReq["required"].([]any)
	if !ok || len(reqSlice) != 1 || reqSlice[0] != "id" {
		t.Errorf("expected required to contain only ['id'], got %v", sanitizedReq["required"])
	}
}
