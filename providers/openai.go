package providers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type OpenAIProvider struct {
	BaseProvider
	BaseURL string
}

func (p *OpenAIProvider) getBaseURL() string {
	if p.BaseURL != "" {
		return strings.TrimSuffix(p.BaseURL, "/")
	}
	return "https://api.openai.com"
}

func (p *OpenAIProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	key := p.GetNextKey()
	defaultModels := []ModelInfo{
		{
			ID:           "gpt-4o",
			Name:         "GPT-4o",
			ContextSize:  128000,
			Capabilities: []string{"vision", "tools", "json"},
		},
		{
			ID:           "gpt-4o-mini",
			Name:         "GPT-4o Mini",
			ContextSize:  128000,
			Capabilities: []string{"vision", "tools", "json"},
		},
		{
			ID:           "o1",
			Name:         "o1",
			ContextSize:  128000,
			Capabilities: []string{"vision", "tools", "json"},
		},
		{
			ID:           "o3-mini",
			Name:         "o3-mini",
			ContextSize:  128000,
			Capabilities: []string{"tools", "json"},
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
	}

	if key == "" {
		return defaultModels, nil
	}

	url := fmt.Sprintf("%s/v1/models", p.getBaseURL())
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return defaultModels, nil
	}
	req.Header.Set("Authorization", "Bearer "+key)

	resp, err := http.DefaultClient.Do(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		if resp != nil {
			resp.Body.Close()
		}
		return defaultModels, nil
	}
	defer resp.Body.Close()

	var data struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return defaultModels, nil
	}

	var models []ModelInfo
	for _, m := range data.Data {
		// Filter for chat / completion models
		if strings.HasPrefix(m.ID, "gpt-") || strings.HasPrefix(m.ID, "o1") || strings.HasPrefix(m.ID, "o3") || strings.HasPrefix(m.ID, "chatgpt-") {
			models = append(models, ModelInfo{
				ID:           m.ID,
				Name:         m.ID,
				ContextSize:  128000,
				Capabilities: []string{"vision", "tools", "json"},
			})
		}
	}
	if len(models) == 0 {
		return defaultModels, nil
	}
	return models, nil
}

func (p *OpenAIProvider) Chat(ctx context.Context, req *CompletionRequest, onChunk func(*CompletionResponse)) (*CompletionResponse, error) {
	key := p.GetNextKey()
	if key == "" {
		return nil, fmt.Errorf("no openai api key configured")
	}

	// Prepare OpenAI messages
	var openAIMessages []map[string]any

	if req.SystemInstruction != "" {
		openAIMessages = append(openAIMessages, map[string]any{
			"role":    "system",
			"content": req.SystemInstruction,
		})
	}

	for _, msg := range req.Messages {
		m := map[string]any{
			"role": msg.Role,
		}
		if msg.Name != "" {
			m["name"] = msg.Name
		}
		if msg.ToolCallID != "" {
			m["tool_call_id"] = msg.ToolCallID
		}

		// Handle ToolCalls
		if len(msg.ToolCalls) > 0 {
			var tcs []map[string]any
			for _, tc := range msg.ToolCalls {
				tcs = append(tcs, map[string]any{
					"id":   tc.ID,
					"type": "function",
					"function": map[string]any{
						"name":      tc.Function.Name,
						"arguments": tc.Function.Arguments,
					},
				})
			}
			m["tool_calls"] = tcs
		}

		// Handle content parts
		hasImages := false
		for _, part := range msg.Content {
			if part.Type == ContentTypeImage || part.Type == ContentTypeDocument {
				hasImages = true
				break
			}
		}

		if hasImages || len(msg.Content) > 1 {
			var parts []map[string]any
			for _, part := range msg.Content {
				switch part.Type {
				case ContentTypeText:
					parts = append(parts, map[string]any{
						"type": "text",
						"text": part.Text,
					})
				case ContentTypeImage, ContentTypeDocument:
					mimeType := part.MimeType
					if mimeType == "" {
						mimeType = "image/jpeg"
					}
					var imageURL string
					if len(part.Data) > 0 {
						imageURL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(part.Data))
					} else if part.FileURI != "" {
						imageURL = part.FileURI
					}
					if imageURL != "" {
						parts = append(parts, map[string]any{
							"type": "image_url",
							"image_url": map[string]any{
								"url": imageURL,
							},
						})
					}
				}
			}
			m["content"] = parts
		} else if len(msg.Content) == 1 {
			m["content"] = msg.Content[0].Text
		} else if len(msg.ToolCalls) > 0 {
			m["content"] = nil
		} else {
			m["content"] = ""
		}

		openAIMessages = append(openAIMessages, m)
	}

	// Prepare OpenAI tools
	var openAITools []map[string]any
	for _, tool := range req.Tools {
		funcs := tool.Functions
		if len(funcs) == 0 && tool.Name != "" {
			funcs = []Tool{tool}
		}
		for _, f := range funcs {
			params := f.Parameters
			if params == nil {
				params = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			openAITools = append(openAITools, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        f.Name,
					"description": f.Description,
					"parameters":  params,
				},
			})
		}
	}

	bodyMap := map[string]any{
		"model":    req.Model,
		"messages": openAIMessages,
		"stream":   req.Stream,
	}
	if len(openAITools) > 0 {
		bodyMap["tools"] = openAITools
	}

	bodyBytes, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("%s/v1/chat/completions", p.getBaseURL())
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+key)

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai api error (status %d): %s", resp.StatusCode, string(respBody))
	}

	if req.Stream {
		var fullContent strings.Builder
		var allToolCalls []ToolCall
		toolCallsMap := make(map[int]*ToolCall)

		reader := bufio.NewReader(resp.Body)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					break
				}
				return nil, err
			}
			line = strings.TrimSpace(line)
			if line == "" || !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}

			var chunk struct {
				Choices []struct {
					Delta struct {
						Content   string `json:"content"`
						ToolCalls []struct {
							Index    int    `json:"index"`
							ID       string `json:"id"`
							Type     string `json:"type"`
							Function struct {
								Name      string `json:"name"`
								Arguments string `json:"arguments"`
							} `json:"function"`
						} `json:"tool_calls"`
					} `json:"delta"`
					FinishReason string `json:"finish_reason"`
				} `json:"choices"`
			}

			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}

			for _, choice := range chunk.Choices {
				if choice.Delta.Content != "" {
					fullContent.WriteString(choice.Delta.Content)
					if onChunk != nil {
						onChunk(&CompletionResponse{Content: choice.Delta.Content})
					}
				}
				for _, tc := range choice.Delta.ToolCalls {
					existing, ok := toolCallsMap[tc.Index]
					if !ok {
						toolCall := &ToolCall{
							ID:   tc.ID,
							Type: "function",
							Function: FunctionCall{
								Name:      tc.Function.Name,
								Arguments: tc.Function.Arguments,
							},
						}
						toolCallsMap[tc.Index] = toolCall
						if onChunk != nil {
							onChunk(&CompletionResponse{ToolCalls: []ToolCall{*toolCall}})
						}
					} else {
						if tc.ID != "" {
							existing.ID = tc.ID
						}
						if tc.Function.Name != "" {
							existing.Function.Name += tc.Function.Name
						}
						if tc.Function.Arguments != "" {
							existing.Function.Arguments += tc.Function.Arguments
						}
					}
				}
			}
		}

		for i := 0; i < len(toolCallsMap); i++ {
			if tc, ok := toolCallsMap[i]; ok {
				allToolCalls = append(allToolCalls, *tc)
			}
		}

		return &CompletionResponse{
			Content:   fullContent.String(),
			ToolCalls: allToolCalls,
		}, nil
	}

	var chatResp struct {
		Choices []struct {
			Message struct {
				Role      string `json:"role"`
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("error decoding openai response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return &CompletionResponse{}, nil
	}

	choice := chatResp.Choices[0]
	var toolCalls []ToolCall
	for _, tc := range choice.Message.ToolCalls {
		toolCalls = append(toolCalls, ToolCall{
			ID:   tc.ID,
			Type: "function",
			Function: FunctionCall{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		})
	}

	return &CompletionResponse{
		Content:   choice.Message.Content,
		ToolCalls: toolCalls,
	}, nil
}
