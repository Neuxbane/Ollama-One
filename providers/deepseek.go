package providers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

// SupportedDeepSeekModels lists the default supported DeepSeek models
var SupportedDeepSeekModels = []ModelInfo{
	{
		ID:           "deepseek/deepseek-chat",
		Name:         "DeepSeek Chat (V3)",
		ContextSize:  64000,
		Capabilities: []string{"tools", "json", "chat", "completion"},
	},
	{
		ID:           "deepseek/deepseek-reasoner",
		Name:         "DeepSeek Reasoner (R1)",
		ContextSize:  64000,
		Capabilities: []string{"reasoning", "tools", "json", "chat", "completion"},
	},
	{
		ID:           "deepseek-chat",
		Name:         "DeepSeek Chat (V3)",
		ContextSize:  64000,
		Capabilities: []string{"tools", "json", "chat", "completion"},
	},
	{
		ID:           "deepseek-reasoner",
		Name:         "DeepSeek Reasoner (R1)",
		ContextSize:  64000,
		Capabilities: []string{"reasoning", "tools", "json", "chat", "completion"},
	},
}

// DeepSeekProvider handles DeepSeek API completions in OpenAI format
type DeepSeekProvider struct {
	BaseProvider
	BaseURL    string
	HTTPClient *http.Client
}

// NewDeepSeekProvider creates a new DeepSeek provider instance
func NewDeepSeekProvider(apiKeys []string) *DeepSeekProvider {
	return &DeepSeekProvider{
		BaseProvider: BaseProvider{APIKeys: apiKeys},
		HTTPClient:   &http.Client{Timeout: 0},
	}
}

func (p *DeepSeekProvider) getBaseURL() string {
	if p.BaseURL != "" {
		return strings.TrimSuffix(p.BaseURL, "/")
	}
	if envURL := os.Getenv("DEEPSEEK_BASE_URL"); envURL != "" {
		return strings.TrimSuffix(envURL, "/")
	}
	return "https://api.deepseek.com"
}

func (p *DeepSeekProvider) getEndpoint(path string) string {
	base := p.getBaseURL()
	path = "/" + strings.TrimPrefix(path, "/")
	return base + path
}

func (p *DeepSeekProvider) getDeepSeekMaxRetries() int {
	if val := os.Getenv("DEEPSEEK_MAX_RETRIES"); val != "" {
		if n, err := strconv.Atoi(val); err == nil && n >= 1 {
			return n
		}
	}
	return getMaxRetries()
}

func (p *DeepSeekProvider) getKeyForAttempt(attempt int, reqKey string) string {
	parsedReqKeys := parseKeys(reqKey)
	if len(parsedReqKeys) > 1 {
		return parsedReqKeys[attempt%len(parsedReqKeys)]
	}

	if attempt == 0 {
		if len(parsedReqKeys) == 1 {
			return parsedReqKeys[0]
		}
		if key := p.GetNextKey(); key != "" {
			return key
		}
		if envKey := os.Getenv("DEEPSEEK_API_KEY"); envKey != "" {
			keys := parseKeys(envKey)
			if len(keys) > 0 {
				return keys[0]
			}
		}
		return ""
	}

	// Retry attempts
	if len(p.APIKeys) > 1 {
		return p.GetNextKey()
	}
	if len(parsedReqKeys) == 1 {
		return parsedReqKeys[0]
	}
	if key := p.GetNextKey(); key != "" {
		return key
	}
	if envKey := os.Getenv("DEEPSEEK_API_KEY"); envKey != "" {
		keys := parseKeys(envKey)
		if len(keys) > 0 {
			return keys[attempt%len(keys)]
		}
	}
	return ""
}

func (p *DeepSeekProvider) isRetryableError(err error) bool {
	if err == nil {
		return false
	}
	errStr := strings.ToLower(err.Error())
	if strings.Contains(errStr, "429") ||
		strings.Contains(errStr, "rate limit") ||
		strings.Contains(errStr, "too many requests") ||
		strings.Contains(errStr, "insufficient_quota") {
		return true
	}
	if strings.Contains(errStr, "500") ||
		strings.Contains(errStr, "502") ||
		strings.Contains(errStr, "503") ||
		strings.Contains(errStr, "504") ||
		strings.Contains(errStr, "internal") ||
		strings.Contains(errStr, "unavailable") ||
		strings.Contains(errStr, "bad gateway") ||
		strings.Contains(errStr, "service unavailable") {
		return true
	}
	if strings.Contains(errStr, "eof") ||
		strings.Contains(errStr, "connection reset") ||
		strings.Contains(errStr, "broken pipe") ||
		strings.Contains(errStr, "dial failed") ||
		strings.Contains(errStr, "timeout") {
		return true
	}
	if strings.Contains(errStr, "401") || strings.Contains(errStr, "403") {
		return len(p.APIKeys) > 1
	}
	return false
}

// ListModels fetches available models from the DeepSeek API
func (p *DeepSeekProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	return p.ListModelsWithKey(ctx, "")
}

// ListModelsWithKey fetches available models using a specific API key or configured keys
func (p *DeepSeekProvider) ListModelsWithKey(ctx context.Context, apiKey string) ([]ModelInfo, error) {
	key := p.getKeyForAttempt(0, apiKey)
	if key == "" {
		return SupportedDeepSeekModels, nil
	}

	req, err := http.NewRequestWithContext(ctx, "GET", p.getEndpoint("/models"), nil)
	if err != nil {
		return SupportedDeepSeekModels, nil
	}
	req.Header.Set("Authorization", "Bearer "+key)

	client := p.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		if resp != nil {
			resp.Body.Close()
		}
		return SupportedDeepSeekModels, nil
	}

	var data struct {
		Data []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		resp.Body.Close()
		return SupportedDeepSeekModels, nil
	}
	resp.Body.Close()

	if len(data.Data) == 0 {
		return SupportedDeepSeekModels, nil
	}

	var models []ModelInfo
	seen := make(map[string]bool)

	for _, m := range data.Data {
		caps := []string{"chat", "completion", "tools", "json"}
		if strings.Contains(m.ID, "reasoner") || strings.Contains(m.ID, "r1") {
			caps = append(caps, "reasoning")
		}

		info := ModelInfo{
			ID:           m.ID,
			Name:         "DeepSeek " + m.ID,
			ContextSize:  64000,
			Capabilities: caps,
		}
		models = append(models, info)
		seen[m.ID] = true

		prefixed := "deepseek/" + m.ID
		if !seen[prefixed] {
			models = append(models, ModelInfo{
				ID:           prefixed,
				Name:         "DeepSeek " + m.ID,
				ContextSize:  64000,
				Capabilities: caps,
			})
			seen[prefixed] = true
		}
	}

	return models, nil
}

func (p *DeepSeekProvider) resolveTargetModel(rawModel string, thinking *ThinkingConfig) string {
	model := strings.TrimSpace(rawModel)
	model = strings.TrimPrefix(model, "deepseek/")

	if model == "" || model == "deepseek" {
		if thinking != nil && thinking.IncludeThoughts {
			return "deepseek-reasoner"
		}
		return "deepseek-chat"
	}
	return model
}

func (p *DeepSeekProvider) buildPayload(targetModel string, req *CompletionRequest, stream bool) map[string]any {
	var messages []map[string]any

	if req.SystemInstruction != "" {
		messages = append(messages, map[string]any{
			"role":    "system",
			"content": req.SystemInstruction,
		})
	}

	for _, m := range req.Messages {
		role := m.Role
		if role == "developer" {
			role = "system"
		}

		var textParts []string
		for _, part := range m.Content {
			if part.Type == ContentTypeText && part.Text != "" {
				textParts = append(textParts, part.Text)
			} else if part.Type == ContentTypeImage {
				textParts = append(textParts, "[Image]")
			}
		}
		contentStr := strings.Join(textParts, "\n")

		msgMap := map[string]any{
			"role": role,
		}

		if len(m.ToolCalls) > 0 && role == "assistant" {
			var tcs []map[string]any
			for _, tc := range m.ToolCalls {
				tcType := tc.Type
				if tcType == "" {
					tcType = "function"
				}
				tcs = append(tcs, map[string]any{
					"id":   tc.ID,
					"type": tcType,
					"function": map[string]any{
						"name":      tc.Function.Name,
						"arguments": tc.Function.Arguments,
					},
				})
			}
			msgMap["tool_calls"] = tcs
			if contentStr != "" {
				msgMap["content"] = contentStr
			} else {
				msgMap["content"] = nil
			}
		} else {
			msgMap["content"] = contentStr
		}

		if m.ToolCallID != "" {
			msgMap["tool_call_id"] = m.ToolCallID
		}
		if m.Name != "" {
			msgMap["name"] = m.Name
		}

		messages = append(messages, msgMap)
	}

	payload := map[string]any{
		"model":    targetModel,
		"messages": messages,
		"stream":   stream,
	}

	// DeepSeek-Reasoner does not support tools or temperature/top_p
	isReasoner := strings.Contains(targetModel, "reasoner")

	if !isReasoner {
		var toolsPayload []map[string]any
		for _, tool := range req.Tools {
			if tool.GoogleSearch {
				continue
			}
			if len(tool.Functions) > 0 {
				for _, f := range tool.Functions {
					name := f.Name
					desc := f.Description
					params := f.Parameters
					if f.Function != nil && f.Function.Name != "" {
						name = f.Function.Name
					}
					if params == nil {
						params = map[string]any{"type": "object", "properties": map[string]any{}}
					}
					toolsPayload = append(toolsPayload, map[string]any{
						"type": "function",
						"function": map[string]any{
							"name":        name,
							"description": desc,
							"parameters":  params,
						},
					})
				}
				continue
			}

			name := tool.Name
			desc := tool.Description
			params := tool.Parameters
			if tool.Function != nil && tool.Function.Name != "" {
				name = tool.Function.Name
			}
			if params == nil {
				params = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			toolsPayload = append(toolsPayload, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        name,
					"description": desc,
					"parameters":  params,
				},
			})
		}

		if len(toolsPayload) > 0 {
			payload["tools"] = toolsPayload
		}

		if req.Temperature != nil {
			payload["temperature"] = *req.Temperature
		}
		if req.TopP != nil {
			payload["top_p"] = *req.TopP
		}
	}

	if req.MaxTokens != nil {
		payload["max_tokens"] = *req.MaxTokens
	}

	return payload
}

// Chat executes a completion request against DeepSeek
func (p *DeepSeekProvider) Chat(ctx context.Context, req *CompletionRequest, onChunk func(*CompletionResponse)) (*CompletionResponse, error) {
	if PreChatHook != nil {
		if err := PreChatHook(req); err != nil {
			return nil, err
		}
	}

	targetModel := p.resolveTargetModel(req.Model, req.Thinking)
	maxRetries := p.getDeepSeekMaxRetries()
	var lastErr error

	for attempt := 0; attempt < maxRetries; attempt++ {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		key := p.getKeyForAttempt(attempt, req.APIKey)
		if key == "" {
			return nil, fmt.Errorf("no DeepSeek API key provided in request (via 'Authorization: Bearer <key>', 'x-api-key', or '?key=<key>') or in env DEEPSEEK_API_KEY")
		}

		chunksYielded := 0
		safeChunkHandler := func(chunk *CompletionResponse) {
			if chunk != nil && (chunk.Content != "" || len(chunk.ToolCalls) > 0 || chunk.Thought != "") {
				chunksYielded++
				if onChunk != nil {
					onChunk(chunk)
				}
			}
		}

		var resp *CompletionResponse
		var err error
		if req.Stream && onChunk != nil {
			resp, err = p.chatStream(ctx, key, targetModel, req, safeChunkHandler)
		} else {
			resp, err = p.chatNonStream(ctx, key, targetModel, req)
		}

		if err == nil && resp != nil {
			if resp.Content != "" || len(resp.ToolCalls) > 0 || resp.Thought != "" {
				return resp, nil
			}
			err = fmt.Errorf("empty response received from DeepSeek (0 content, 0 tool calls)")
		}

		lastErr = err

		if chunksYielded > 0 {
			log.Printf("[WARN][DEEPSEEK][CHAT] Error occurred after streaming chunks yielded (%d chunks): %v", chunksYielded, err)
			return resp, err
		}

		if !p.isRetryableError(err) {
			break
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(time.Duration(1<<attempt) * 200 * time.Millisecond):
		}
	}

	return nil, lastErr
}

func (p *DeepSeekProvider) chatNonStream(ctx context.Context, key string, targetModel string, req *CompletionRequest) (*CompletionResponse, error) {
	payload := p.buildPayload(targetModel, req, false)
	jsonBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal deepseek request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.getEndpoint("/chat/completions"), bytes.NewReader(jsonBytes))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+key)
	httpReq.Header.Set("Accept", "application/json")

	client := p.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read deepseek response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("deepseek api error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	var dsResp struct {
		ID      string `json:"id"`
		Choices []struct {
			Index   int `json:"index"`
			Message struct {
				Role             string `json:"role"`
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
				ToolCalls        []struct {
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

	if err := json.Unmarshal(bodyBytes, &dsResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal deepseek response: %w", err)
	}

	if len(dsResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices returned by deepseek")
	}

	choice := dsResp.Choices[0]
	result := &CompletionResponse{
		Content: choice.Message.Content,
		Thought: choice.Message.ReasoningContent,
	}

	for _, tc := range choice.Message.ToolCalls {
		tcType := tc.Type
		if tcType == "" {
			tcType = "function"
		}
		result.ToolCalls = append(result.ToolCalls, ToolCall{
			ID:   tc.ID,
			Type: tcType,
			Function: FunctionCall{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		})
	}

	return result, nil
}

func (p *DeepSeekProvider) chatStream(ctx context.Context, key string, targetModel string, req *CompletionRequest, onChunk func(*CompletionResponse)) (*CompletionResponse, error) {
	payload := p.buildPayload(targetModel, req, true)
	jsonBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal deepseek request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.getEndpoint("/chat/completions"), bytes.NewReader(jsonBytes))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+key)
	httpReq.Header.Set("Accept", "text/event-stream")

	client := p.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("deepseek api error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	reader := bufio.NewReader(resp.Body)
	var fullContent strings.Builder
	var fullThought strings.Builder

	type pendingTC struct {
		ID        string
		Type      string
		Name      string
		Arguments strings.Builder
	}
	pendingToolCalls := make(map[int]*pendingTC)

	for {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return nil, err
		}

		trimmed := strings.TrimSpace(line)
		if trimmed != "" && strings.HasPrefix(trimmed, "data:") {
			data := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
			if data == "[DONE]" {
				break
			}

			var chunk struct {
				Choices []struct {
					Index int `json:"index"`
					Delta struct {
						Role             string `json:"role"`
						Content          string `json:"content"`
						ReasoningContent string `json:"reasoning_content"`
						ToolCalls        []struct {
							Index    int    `json:"index"`
							ID       string `json:"id"`
							Type     string `json:"type"`
							Function struct {
								Name      string `json:"name"`
								Arguments string `json:"arguments"`
							} `json:"function"`
						} `json:"tool_calls"`
					} `json:"delta"`
					FinishReason *string `json:"finish_reason"`
				} `json:"choices"`
			}

			if err := json.Unmarshal([]byte(data), &chunk); err == nil {
				if len(chunk.Choices) > 0 {
					choice := chunk.Choices[0]

					if choice.Delta.ReasoningContent != "" {
						fullThought.WriteString(choice.Delta.ReasoningContent)
						if onChunk != nil {
							onChunk(&CompletionResponse{Thought: choice.Delta.ReasoningContent})
						}
					}

					if choice.Delta.Content != "" {
						fullContent.WriteString(choice.Delta.Content)
						if onChunk != nil {
							onChunk(&CompletionResponse{Content: choice.Delta.Content})
						}
					}

					for _, tc := range choice.Delta.ToolCalls {
						ptc, exists := pendingToolCalls[tc.Index]
						if !exists {
							ptc = &pendingTC{}
							pendingToolCalls[tc.Index] = ptc
						}
						if tc.ID != "" {
							ptc.ID = tc.ID
						}
						if tc.Type != "" {
							ptc.Type = tc.Type
						}
						if tc.Function.Name != "" {
							ptc.Name = tc.Function.Name
						}
						if tc.Function.Arguments != "" {
							ptc.Arguments.WriteString(tc.Function.Arguments)
						}
					}
				}
			}
		}

		if err == io.EOF {
			break
		}
	}

	var indices []int
	for idx := range pendingToolCalls {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	var finalToolCalls []ToolCall
	for _, idx := range indices {
		ptc := pendingToolCalls[idx]
		tcType := ptc.Type
		if tcType == "" {
			tcType = "function"
		}
		finalToolCalls = append(finalToolCalls, ToolCall{
			ID:   ptc.ID,
			Type: tcType,
			Function: FunctionCall{
				Name:      ptc.Name,
				Arguments: ptc.Arguments.String(),
			},
		})
	}

	if len(finalToolCalls) > 0 && onChunk != nil {
		onChunk(&CompletionResponse{ToolCalls: finalToolCalls})
	}

	return &CompletionResponse{
		Content:   fullContent.String(),
		Thought:   fullThought.String(),
		ToolCalls: finalToolCalls,
	}, nil
}
