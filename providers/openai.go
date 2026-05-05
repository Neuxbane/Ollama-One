package providers

import (
	"context"
	"fmt"
)

type OpenAIProvider struct {
	BaseProvider
}

type openAIResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func (p *OpenAIProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	// OpenAI doesn't have a simple way to list only "chat" models without filtering
	// For now, return a static list of popular models if keys are present
	if len(p.APIKeys) == 0 {
		return nil, fmt.Errorf("no api keys configured")
	}

	return []ModelInfo{
		{
			ID:           "gpt-4o",
			Name:         "GPT-4o",
			ContextSize:  128000,
			Capabilities: []string{"vision", "tools", "json"},
		},
		{
			ID:           "gpt-4-turbo",
			Name:         "GPT-4 Turbo",
			ContextSize:  128000,
			Capabilities: []string{"vision", "tools", "json"},
		},
		{
			ID:           "gpt-3.5-turbo",
			Name:         "GPT-3.5 Turbo",
			ContextSize:  16385,
			Capabilities: []string{"tools", "json"},
		},
	}, nil
}

func (p *OpenAIProvider) Chat(ctx context.Context, req *CompletionRequest, onChunk func(*CompletionResponse)) (*CompletionResponse, error) {
	// Implement basic non-streaming for now as a placeholder
	// If the user wants full OpenAI support, we'd add SSE parsing here too
	return &CompletionResponse{
		Content: "OpenAI response placeholder (streaming not yet implemented for OpenAI)",
	}, nil
}
