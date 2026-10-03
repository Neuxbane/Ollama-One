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
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// SupportedLiveModels lists the known Gemini Live models supported over WebSocket
var SupportedLiveModels = []ModelInfo{
	{
		ID:           "google/ws/gemini-3.8-live-extended-thinking",
		Name:         "Gemini 3.8 Flash Live Extended Thinking (WebSocket)",
		ContextSize:  128000,
		Capabilities: []string{"audio", "vision", "tools", "chat", "completion"},
	},
	{
		ID:           "google/ws/gemini-3.8-live",
		Name:         "Gemini 3.8 Flash Live (WebSocket)",
		ContextSize:  128000,
		Capabilities: []string{"audio", "vision", "tools", "chat", "completion"},
	},
	{
		ID:           "google/ws/gemini-3.1-flash-live-preview",
		Name:         "Gemini 3.1 Flash Live Preview (WebSocket)",
		ContextSize:  128000,
		Capabilities: []string{"audio", "vision", "tools", "chat", "completion"},
	},
	{
		ID:           "google/ws/gemini-2.5-flash-native-audio-preview-09-2025",
		Name:         "Gemini 2.5 Flash Native Audio Preview 09-2025 (WebSocket)",
		ContextSize:  128000,
		Capabilities: []string{"audio", "vision", "tools", "chat", "completion"},
	},
	{
		ID:           "google/ws/gemini-2.5-flash-native-audio-preview-12-2025",
		Name:         "Gemini 2.5 Flash Native Audio Preview 12-2025 (WebSocket)",
		ContextSize:  128000,
		Capabilities: []string{"audio", "vision", "tools", "chat", "completion"},
	},
}

// GoogleProvider handles both Google Gemini REST API and WebSocket Live Bidi API
type GoogleProvider struct {
	BaseProvider
	BaseURL            string
	WSURL              string
	ResponseModalities []string
}

// NewGoogleProvider creates a new Google provider instance
func NewGoogleProvider(apiKeys []string) *GoogleProvider {
	return &GoogleProvider{
		BaseProvider:       BaseProvider{APIKeys: apiKeys},
		ResponseModalities: []string{"AUDIO"},
	}
}

func (p *GoogleProvider) getBaseURL() string {
	if p.BaseURL != "" {
		return strings.TrimSuffix(p.BaseURL, "/")
	}
	return "https://generativelanguage.googleapis.com"
}

// ListModels fetches available models from the Google API and appends WebSocket live models
func (p *GoogleProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	return p.ListModelsWithKey(ctx, "")
}

// ListModelsWithKey fetches available models using a specific API key (if provided) or configured keys
func (p *GoogleProvider) ListModelsWithKey(ctx context.Context, apiKey string) ([]ModelInfo, error) {
	key := apiKey
	if key == "" {
		key = p.GetNextKey()
		if key == "" {
			key = os.Getenv("GEMINI_API_KEY")
			if key == "" {
				key = os.Getenv("GOOGLE_API_KEY")
			}
		}
	}

	var models []ModelInfo
	// Add live models first
	models = append(models, SupportedLiveModels...)

	if key == "" {
		// In stateless mode without a key, return statically known supported models
		return models, nil
	}

	pageToken := ""
	for {
		urlStr := fmt.Sprintf("%s/v1beta/models?key=%s", p.getBaseURL(), key)
		if pageToken != "" {
			urlStr += fmt.Sprintf("&pageToken=%s", pageToken)
		}

		req, err := http.NewRequestWithContext(ctx, "GET", urlStr, nil)
		if err != nil {
			return models, nil
		}

		resp, err := http.DefaultClient.Do(req)
		if err != nil || resp.StatusCode != http.StatusOK {
			if resp != nil {
				resp.Body.Close()
			}
			return models, nil
		}

		var data struct {
			Models []struct {
				Name             string   `json:"name"`
				DisplayName      string   `json:"displayName"`
				InputTokenLimit  int      `json:"inputTokenLimit"`
				Thinking         bool     `json:"thinking"`
				SupportedMethods []string `json:"supportedGenerationMethods"`
			} `json:"models"`
			NextPageToken string `json:"nextPageToken"`
		}

		if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
			resp.Body.Close()
			return models, nil
		}
		resp.Body.Close()

		for _, m := range data.Models {
			supportsGenerate := false
			for _, method := range m.SupportedMethods {
				if method == "generateContent" {
					supportsGenerate = true
					break
				}
			}
			if !supportsGenerate {
				continue
			}

			id := strings.TrimPrefix(m.Name, "models/")
			caps := []string{"vision", "tools", "chat", "completion"}
			if m.Thinking {
				caps = append(caps, "thinking")
			}

			// Add with google/ prefix
			models = append(models, ModelInfo{
				ID:           "google/" + id,
				Name:         m.DisplayName,
				ContextSize:  m.InputTokenLimit,
				Capabilities: caps,
			})
		}

		if data.NextPageToken == "" {
			break
		}
		pageToken = data.NextPageToken
	}

	return models, nil
}

func getMaxRetries() int {
	if val := os.Getenv("GEMINI_MAX_RETRIES"); val != "" {
		if n, err := strconv.Atoi(val); err == nil && n >= 1 {
			return n
		}
	}
	return 3
}

func parseKeys(raw string) []string {
	if raw == "" {
		return nil
	}
	var keys []string
	for _, k := range strings.Split(raw, ",") {
		k = strings.TrimSpace(k)
		if k != "" {
			keys = append(keys, k)
		}
	}
	return keys
}

func (p *GoogleProvider) getKeyForAttempt(attempt int, reqKey string) string {
	parsedReqKeys := parseKeys(reqKey)
	if len(parsedReqKeys) > 1 {
		return parsedReqKeys[attempt%len(parsedReqKeys)]
	}

	if attempt == 0 {
		if len(parsedReqKeys) == 1 {
			return parsedReqKeys[0]
		}
		key := p.GetNextKey()
		if key != "" {
			return key
		}
		if envKey := os.Getenv("GEMINI_API_KEY"); envKey != "" {
			keys := parseKeys(envKey)
			if len(keys) > 0 {
				return keys[0]
			}
		}
		if envKey := os.Getenv("GOOGLE_API_KEY"); envKey != "" {
			keys := parseKeys(envKey)
			if len(keys) > 0 {
				return keys[0]
			}
		}
		return ""
	}

	// Retry attempts (attempt > 0)
	if len(p.APIKeys) > 1 {
		return p.GetNextKey()
	}
	if len(parsedReqKeys) == 1 {
		return parsedReqKeys[0]
	}
	key := p.GetNextKey()
	if key != "" {
		return key
	}
	if envKey := os.Getenv("GEMINI_API_KEY"); envKey != "" {
		keys := parseKeys(envKey)
		if len(keys) > 0 {
			return keys[attempt%len(keys)]
		}
	}
	if envKey := os.Getenv("GOOGLE_API_KEY"); envKey != "" {
		keys := parseKeys(envKey)
		if len(keys) > 0 {
			return keys[attempt%len(keys)]
		}
	}
	return ""
}

func (p *GoogleProvider) isRetryableError(err error) bool {
	if err == nil {
		return false
	}
	errStr := strings.ToLower(err.Error())
	if strings.Contains(errStr, "429") ||
		strings.Contains(errStr, "resource_exhausted") ||
		strings.Contains(errStr, "quota") ||
		strings.Contains(errStr, "rate limit") ||
		strings.Contains(errStr, "too many requests") {
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
		strings.Contains(errStr, "close 1000") ||
		strings.Contains(errStr, "close 1006") ||
		strings.Contains(errStr, "closed network connection") ||
		strings.Contains(errStr, "websocket: close") ||
		strings.Contains(errStr, "empty response") ||
		strings.Contains(errStr, "timeout") {
		return true
	}
	if strings.Contains(errStr, "401") || strings.Contains(errStr, "403") {
		return len(p.APIKeys) > 1
	}
	return false
}

// Chat routes the request strictly:
// - "google/ws/<model>" routes to WebSocket Live
// - "google/<model>" (or default) routes to Google REST API
// It performs auto-retries on transient errors, rate limits (429), server errors (5xx), and empty responses.
func (p *GoogleProvider) Chat(ctx context.Context, req *CompletionRequest, onChunk func(*CompletionResponse)) (*CompletionResponse, error) {
	if PreChatHook != nil {
		if err := PreChatHook(req); err != nil {
			return nil, err
		}
	}

	maxRetries := getMaxRetries()
	var lastErr error

	for attempt := 0; attempt < maxRetries; attempt++ {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		key := p.getKeyForAttempt(attempt, req.APIKey)
		if key == "" {
			return nil, fmt.Errorf("no Google API key provided in request (via 'Authorization: Bearer <key>', 'x-api-key', or '?key=<key>') or in config/env")
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

		rawModel := strings.TrimSpace(req.Model)
		var resp *CompletionResponse
		var err error

		if strings.HasPrefix(rawModel, "google/ws/") {
			targetModel := strings.TrimPrefix(rawModel, "google/ws/")
			resp, err = p.chatWebSocket(ctx, key, targetModel, req, safeChunkHandler)
		} else {
			targetModel := strings.TrimPrefix(rawModel, "google/")
			resp, err = p.chatREST(ctx, key, targetModel, req, safeChunkHandler)
		}

		// Check if attempt succeeded
		if err == nil && resp != nil {
			if resp.Content != "" || len(resp.ToolCalls) > 0 {
				return resp, nil
			}
			err = fmt.Errorf("empty response received from Gemini (0 content, 0 tool calls)")
		}

		lastErr = err

		// If chunks have already been emitted to the client, we cannot safely replay without duplication
		if chunksYielded > 0 {
			log.Printf("[WARN][CHAT] Error occurred after streaming chunks yielded (%d chunks): %v", chunksYielded, err)
			return resp, err
		}

		// Check if we should retry
		if attempt < maxRetries-1 && p.isRetryableError(err) {
			backoff := time.Duration(1<<attempt) * 500 * time.Millisecond
			if strings.Contains(strings.ToLower(err.Error()), "429") ||
				strings.Contains(strings.ToLower(err.Error()), "resource_exhausted") ||
				strings.Contains(strings.ToLower(err.Error()), "quota") {
				backoff = time.Duration(attempt+1) * 1000 * time.Millisecond
			}

			log.Printf("[RETRY] Attempt %d/%d failed: %v. Retrying in %v...", attempt+1, maxRetries, err, backoff)

			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
			continue
		}

		// Not retryable or max attempts reached
		break
	}

	return nil, lastErr
}

// --- Google REST API Implementation ---

type geminiFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type geminiFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
	ID   string         `json:"id,omitempty"`
}

type geminiFunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
	ID       string         `json:"id,omitempty"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	Thought          bool                    `json:"thought,omitempty"`
	ThoughtSignature string                  `json:"thoughtSignature,omitempty"`
	InlineData       *geminiInline           `json:"inlineData,omitempty"`
	FileData         *geminiFileData         `json:"fileData,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
}

type geminiInline struct {
	MimeType string `json:"mimeType"`
	Data     []byte `json:"data"`
}

type geminiFileData struct {
	MimeType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiGenerationConfig struct {
	ThinkingConfig *ThinkingConfig `json:"thinkingConfig,omitempty"`
}

type geminiTool struct {
	GoogleSearch         map[string]any   `json:"googleSearch,omitempty"`
	FunctionDeclarations []geminiFunction `json:"functionDeclarations,omitempty"`
}

type geminiSafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

type geminiRequest struct {
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	Contents          []geminiContent         `json:"contents"`
	Tools             []geminiTool            `json:"tools,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
	SafetySettings    []geminiSafetySetting  `json:"safetySettings,omitempty"`
}

type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Role  string       `json:"role"`
			Parts []geminiPart `json:"parts"`
		} `json:"content"`
		FinishReason  string `json:"finishReason"`
		FinishMessage string `json:"finishMessage"`
	} `json:"candidates"`
}

var (
	thoughtSignaturesLock sync.RWMutex
	thoughtSignatures     = make(map[string]string)
)

func cacheThoughtSignature(id string, name string, sig string) {
	if sig == "" {
		return
	}
	thoughtSignaturesLock.Lock()
	defer thoughtSignaturesLock.Unlock()
	if id != "" {
		thoughtSignatures[id] = sig
	}
	if name != "" {
		thoughtSignatures[name] = sig
	}
}

func getThoughtSignature(id string, name string) string {
	thoughtSignaturesLock.RLock()
	defer thoughtSignaturesLock.RUnlock()
	if id != "" {
		if sig, ok := thoughtSignatures[id]; ok {
			return sig
		}
	}
	if name != "" {
		if sig, ok := thoughtSignatures[name]; ok {
			return sig
		}
	}
	return ""
}

func (p *GoogleProvider) buildGeminiContents(messages []Message, isLive bool) ([]geminiContent, []string) {
	var systemInstructions []string
	var chatMessages []Message
	for _, msg := range messages {
		if msg.Role == "system" || msg.Role == "developer" {
			for _, part := range msg.Content {
				if part.Text != "" {
					systemInstructions = append(systemInstructions, part.Text)
				}
			}
		} else {
			chatMessages = append(chatMessages, msg)
		}
	}

	var contents []geminiContent
	for i, msg := range chatMessages {
		role := msg.Role
		switch role {
		case "assistant":
			role = "model"
		case "tool":
			role = "user"
		default:
			role = "user"
		}

		var parts []geminiPart

		if isLive && len(msg.ToolCalls) > 0 {
			var bulkCalls []map[string]any
			for _, tc := range msg.ToolCalls {
				var args any
				if tc.Function.Arguments != "" {
					var parsed map[string]any
					if err := json.Unmarshal([]byte(tc.Function.Arguments), &parsed); err == nil {
						args = parsed
					} else {
						args = tc.Function.Arguments
					}
				} else {
					args = map[string]any{}
				}
				bulkCalls = append(bulkCalls, map[string]any{
					"name":      tc.Function.Name,
					"arguments": args,
				})
			}
			bulkJSON, _ := json.Marshal(bulkCalls)
			parts = append(parts, geminiPart{
				Text: fmt.Sprintf("<tool_calls>\n%s\n</tool_calls>", string(bulkJSON)),
			})
		} else {
			for _, tc := range msg.ToolCalls {
				var args map[string]any
				if tc.Function.Arguments != "" {
					if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
						args = make(map[string]any)
					}
				} else {
					args = make(map[string]any)
				}
				sig := tc.ThoughtSignature
				if sig == "" {
					sig = getThoughtSignature(tc.ID, tc.Function.Name)
				}
				parts = append(parts, geminiPart{
					FunctionCall: &geminiFunctionCall{
						Name: tc.Function.Name,
						Args: args,
						ID:   tc.ID,
					},
					ThoughtSignature: sig,
				})
			}
		}

		funcName := msg.Name
		if msg.Role == "tool" || msg.Role == "function" {
			if funcName == "" && msg.ToolCallID != "" {
				for j := i - 1; j >= 0; j-- {
					for _, tc := range chatMessages[j].ToolCalls {
						if tc.ID == msg.ToolCallID {
							funcName = tc.Function.Name
							break
						}
					}
					if funcName != "" {
						break
					}
				}
			}
			if funcName == "" {
				for j := i - 1; j >= 0; j-- {
					if len(chatMessages[j].ToolCalls) > 0 {
						funcName = chatMessages[j].ToolCalls[0].Function.Name
						break
					}
				}
			}
			for _, pfx := range []string{"tool_use:", "tool_code:", "tool_call:", "call:"} {
				funcName = strings.TrimPrefix(funcName, pfx)
			}
			if funcName == "" {
				funcName = "unknown"
			}
		}

		for _, part := range msg.Content {
			gp := geminiPart{}
			switch part.Type {
			case ContentTypeText:
				if msg.Role == "tool" || msg.Role == "function" {
					toolText := part.Text
					if isLive {
						var parsedResp any = toolText
						var jsonResult any
						if err := json.Unmarshal([]byte(toolText), &jsonResult); err == nil {
							parsedResp = jsonResult
						}
						respArray := []map[string]any{
							{
								"name":     funcName,
								"response": parsedResp,
							},
						}
						b, _ := json.Marshal(respArray)
						gp.Text = fmt.Sprintf("<tool_responses>\n%s\n</tool_responses>", string(b))
						log.Printf("[DEBUG][WS -> GEMINI] History Tool Result as standardized JSON | Tool: %s, ID: %s", funcName, msg.ToolCallID)
					} else {
						respMap := map[string]any{"result": toolText}
						var jsonResult any
						if err := json.Unmarshal([]byte(part.Text), &jsonResult); err == nil {
							if m, ok := jsonResult.(map[string]any); ok {
								respMap = m
							} else {
								respMap = map[string]any{"result": jsonResult}
							}
						}
						gp.FunctionResponse = &geminiFunctionResponse{
							Name:     funcName,
							Response: respMap,
							ID:       msg.ToolCallID,
						}
						respJSON, _ := json.Marshal(respMap)
						respPreview := string(respJSON)
						if len(respPreview) > 300 {
							respPreview = respPreview[:300] + "..."
						}
						log.Printf("[DEBUG][REST -> GEMINI] Function Response | Tool: %s, ID: %s, Response: %s", funcName, msg.ToolCallID, respPreview)
					}
				} else {
					gp.Text = part.Text
				}
			case ContentTypeImage, ContentTypeDocument:
				if part.FileURI != "" {
					gp.FileData = &geminiFileData{
						MimeType: part.MimeType,
						FileURI:  part.FileURI,
					}
				} else {
					gp.InlineData = &geminiInline{
						MimeType: part.MimeType,
						Data:     part.Data,
					}
				}
			}
			if gp.Text != "" || gp.InlineData != nil || gp.FileData != nil || gp.FunctionResponse != nil {
				parts = append(parts, gp)
			}
		}

		if len(parts) == 0 {
			continue
		}

		if len(contents) > 0 && contents[len(contents)-1].Role == role {
			contents[len(contents)-1].Parts = append(contents[len(contents)-1].Parts, parts...)
		} else {
			contents = append(contents, geminiContent{
				Role:  role,
				Parts: parts,
			})
		}
	}

	return contents, systemInstructions
}

func (p *GoogleProvider) chatREST(ctx context.Context, key string, model string, req *CompletionRequest, onChunk func(*CompletionResponse)) (*CompletionResponse, error) {
	contents, systemInstructions := p.buildGeminiContents(req.Messages, false)
	if req.SystemInstruction != "" {
		systemInstructions = append([]string{req.SystemInstruction}, systemInstructions...)
	}

	gemReq := geminiRequest{
		Contents: contents,
		SafetySettings: []geminiSafetySetting{
			{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_NONE"},
			{Category: "HARM_CATEGORY_HATE_SPEECH", Threshold: "BLOCK_NONE"},
			{Category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", Threshold: "BLOCK_NONE"},
			{Category: "HARM_CATEGORY_DANGEROUS_CONTENT", Threshold: "BLOCK_NONE"},
			{Category: "HARM_CATEGORY_CIVIC_INTEGRITY", Threshold: "BLOCK_NONE"},
		},
	}

	if len(systemInstructions) > 0 {
		gemReq.SystemInstruction = &geminiContent{
			Parts: []geminiPart{{Text: strings.Join(systemInstructions, "\n\n")}},
		}
	}

	var allFuncs []geminiFunction
	for _, tool := range req.Tools {
		if tool.GoogleSearch {
			gemReq.Tools = append(gemReq.Tools, geminiTool{GoogleSearch: make(map[string]any)})
		}
		for _, f := range tool.Functions {
			allFuncs = append(allFuncs, geminiFunction{
				Name:        f.Name,
				Description: f.Description,
				Parameters:  sanitizeGeminiSchemaMap(f.Parameters),
			})
		}
		if len(tool.Functions) == 0 && tool.Name != "" {
			allFuncs = append(allFuncs, geminiFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  sanitizeGeminiSchemaMap(tool.Parameters),
			})
		}
	}
	if len(allFuncs) > 0 {
		gemReq.Tools = append(gemReq.Tools, geminiTool{FunctionDeclarations: allFuncs})
	}

	if req.Thinking != nil {
		gemReq.GenerationConfig = &geminiGenerationConfig{
			ThinkingConfig: req.Thinking,
		}
	}

	bodyBytes, err := json.Marshal(gemReq)
	if err != nil {
		return nil, err
	}

	urlStr := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse&key=%s", p.getBaseURL(), model, key)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", urlStr, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini api error (status %d): %s", resp.StatusCode, string(body))
	}

	fullContent := ""
	var allToolCalls []ToolCall
	reader := bufio.NewReader(resp.Body)
	inThought := false
	pendingText := ""

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
		var gemResp geminiResponse
		if err := json.Unmarshal([]byte(data), &gemResp); err != nil {
			continue
		}

		for _, cand := range gemResp.Candidates {
			for _, part := range cand.Content.Parts {
				if part.Text != "" {
					textToYield := part.Text
					if part.Thought {
						if !inThought {
							textToYield = "<think>\n" + textToYield
							inThought = true
						}
						fullContent += textToYield
						if onChunk != nil {
							onChunk(&CompletionResponse{Content: textToYield})
						}
					} else {
						if inThought {
							closing := "\n</think>\n\n"
							fullContent += closing
							if onChunk != nil {
								onChunk(&CompletionResponse{Content: closing})
							}
							inThought = false
						}

						pendingText = p.processTextStream(
							pendingText,
							textToYield,
							req.Tools,
							func(tc ToolCall) {
								log.Printf("[DEBUG][GEMINI -> PROXY] REST Parsed Text Tool Call | Func: %s, ID: %s, Args: %s", tc.Function.Name, tc.ID, tc.Function.Arguments)
								allToolCalls = append(allToolCalls, tc)
								if onChunk != nil {
									onChunk(&CompletionResponse{ToolCalls: []ToolCall{tc}})
								}
							},
							func(text string) {
								fullContent += text
								if onChunk != nil {
									onChunk(&CompletionResponse{Content: text})
								}
							},
						)
					}
				}

				if part.FunctionCall != nil {
					if inThought {
						closing := "\n</think>\n\n"
						fullContent += closing
						if onChunk != nil {
							onChunk(&CompletionResponse{Content: closing})
						}
						inThought = false
					}

					funcName := part.FunctionCall.Name
					prefixes := []string{"tool_use:", "tool_code:", "tool_call:", "call:"}
					for _, pfx := range prefixes {
						funcName = strings.TrimPrefix(funcName, pfx)
					}

					rawArgsBytes, _ := json.Marshal(part.FunctionCall.Args)
					fixedArgs, _ := p.fixToolCall(funcName, part.FunctionCall.Args, req.Tools)
					argsBytes, _ := json.Marshal(fixedArgs)

					tcID := part.FunctionCall.ID
					if tcID == "" {
						tcID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), len(allToolCalls))
					}
					cacheThoughtSignature(tcID, funcName, part.ThoughtSignature)
					log.Printf("[DEBUG][GEMINI -> PROXY] REST Tool Call | Func: %s, ID: %s, RawArgs: %s, FixedArgs: %s (sig len: %d)", funcName, tcID, string(rawArgsBytes), string(argsBytes), len(part.ThoughtSignature))
					toolCall := ToolCall{
						ID:   tcID,
						Type: "function",
						Function: FunctionCall{
							Name:      funcName,
							Arguments: string(argsBytes),
						},
						ThoughtSignature: part.ThoughtSignature,
					}
					allToolCalls = append(allToolCalls, toolCall)
					if onChunk != nil {
						onChunk(&CompletionResponse{ToolCalls: []ToolCall{toolCall}})
					}
				}
			}

			if cand.FinishReason == "MALFORMED_FUNCTION_CALL" && cand.FinishMessage != "" {
				re := regexp.MustCompile(`(?s).*?(?:call:)?([a-zA-Z0-9_]+)[\s\n]*\{+(.*)`)
				matches := re.FindStringSubmatch(cand.FinishMessage)
				if len(matches) >= 3 {
					funcName := matches[1]
					argsStr := strings.TrimSpace(matches[2])
					lastBrace := strings.LastIndex(argsStr, "}")
					if lastBrace != -1 {
						argsStr = argsStr[:lastBrace]
					}
					argsStr = strings.TrimSpace(argsStr)
					if !strings.HasPrefix(argsStr, "{") {
						argsStr = "{" + argsStr + "}"
					}
					keyFixer := regexp.MustCompile(`([{,])\s*([a-zA-Z0-9_]+)\s*:`)
					argsStr = keyFixer.ReplaceAllString(argsStr, `$1"$2":`)
					quoteFixer := regexp.MustCompile(`'([^']*)'`)
					argsStr = quoteFixer.ReplaceAllString(argsStr, `"$1"`)

					var args map[string]any
					if err := json.Unmarshal([]byte(argsStr), &args); err == nil {
						fixedArgs, _ := p.fixToolCall(funcName, args, req.Tools)
						finalArgs, _ := json.Marshal(fixedArgs)
						toolCall := ToolCall{
							ID:   fmt.Sprintf("call_malformed_%d", len(allToolCalls)),
							Type: "function",
							Function: FunctionCall{
								Name:      funcName,
								Arguments: string(finalArgs),
							},
						}
						allToolCalls = append(allToolCalls, toolCall)
						if onChunk != nil {
							onChunk(&CompletionResponse{ToolCalls: []ToolCall{toolCall}})
						}
					}
				}
			}
		}
	}

	if inThought {
		closing := "\n</think>\n"
		fullContent += closing
		if onChunk != nil {
			onChunk(&CompletionResponse{Content: closing})
		}
	}
	p.flushTextStream(
		pendingText,
		req.Tools,
		func(tc ToolCall) {
			log.Printf("[DEBUG][GEMINI -> PROXY] REST Flushed Text Tool Call | Func: %s, ID: %s, Args: %s", tc.Function.Name, tc.ID, tc.Function.Arguments)
			allToolCalls = append(allToolCalls, tc)
			if onChunk != nil {
				onChunk(&CompletionResponse{ToolCalls: []ToolCall{tc}})
			}
		},
		func(text string) {
			fullContent += text
			if onChunk != nil {
				onChunk(&CompletionResponse{Content: text})
			}
		},
	)

	return &CompletionResponse{Content: fullContent, ToolCalls: allToolCalls}, nil
}

func (p *GoogleProvider) fixToolCall(funcName string, args map[string]any, availableTools []Tool) (map[string]any, error) {
	if args == nil {
		args = make(map[string]any)
	}

	var targetFunc *Tool
	for i := range availableTools {
		for j := range availableTools[i].Functions {
			if availableTools[i].Functions[j].Name == funcName {
				targetFunc = &availableTools[i].Functions[j]
				break
			}
		}
		if targetFunc != nil {
			break
		}
	}

	if targetFunc == nil {
		return args, nil
	}

	if params, ok := targetFunc.Parameters["properties"].(map[string]any); ok {
		required := []string{}
		if req, ok := targetFunc.Parameters["required"].([]any); ok {
			for _, r := range req {
				if s, ok := r.(string); ok {
					required = append(required, s)
				}
			}
		}

		for _, reqKey := range required {
			if _, exists := args[reqKey]; exists {
				continue
			}
			for argKey, argVal := range args {
				isOtherReq := false
				for _, otherReq := range required {
					if otherReq == argKey {
						isOtherReq = true
						break
					}
				}
				if isOtherReq {
					continue
				}

				lArg := strings.ToLower(argKey)
				lReq := strings.ToLower(reqKey)
				if (lArg == "path" && lReq == "filepath") ||
					(lArg == "content" && lReq == "text") ||
					strings.Contains(lReq, lArg) || strings.Contains(lArg, lReq) {
					args[reqKey] = argVal
					delete(args, argKey)
					log.Printf("[DEBUG][FIX TOOL CALL] %s: mapped parameter '%s' -> '%s' (value: %v)", funcName, argKey, reqKey, argVal)
					break
				}
			}
		}

		for _, reqKey := range required {
			if _, exists := args[reqKey]; exists {
				continue
			}

			propInfo, ok := params[reqKey].(map[string]any)
			if !ok {
				continue
			}
			propType, _ := propInfo["type"].(string)

			switch propType {
			case "string":
				args[reqKey] = ""
			case "integer", "number":
				if strings.Contains(strings.ToLower(reqKey), "line") || strings.Contains(strings.ToLower(reqKey), "start") {
					args[reqKey] = 1
				} else {
					args[reqKey] = 0
				}
			case "boolean":
				args[reqKey] = false
			case "array":
				args[reqKey] = []any{}
			case "object":
				args[reqKey] = map[string]any{}
			}
			log.Printf("[DEBUG][FIX TOOL CALL] %s: injected missing required '%s' = %v (type: %s)", funcName, reqKey, args[reqKey], propType)
		}
	}

	return args, nil
}

type toolTagPattern struct {
	opener string
	closer string
}

var knownToolPatterns = []toolTagPattern{
	{"<tool_calls>", "</tool_calls>"},
	{"<tool_call>", "</tool_call>"},
	{"<tool_code>", "</tool_code>"},
	{"<toolUse>", "</toolUse>"},
	{"<tool_use>", "</tool_use>"},
	{"<function_calls>", "</function_calls>"},
	{"[Called Tool:", "]"},
	{"call:", "}"},
	{"```json", "```"},
}

func cleanJSONString(s string) string {
	s = strings.TrimSpace(s)
	keyFixer := regexp.MustCompile(`([{,])\s*([a-zA-Z0-9_]+)\s*:`)
	s = keyFixer.ReplaceAllString(s, `$1"$2":`)
	quoteFixer := regexp.MustCompile(`'([^']*)'`)
	s = quoteFixer.ReplaceAllString(s, `"$1"`)
	return s
}

func cleanFunctionName(name string) string {
	name = strings.TrimSpace(name)
	prefixes := []string{"tool_use:", "tool_code:", "tool_call:", "call:"}
	for _, pfx := range prefixes {
		name = strings.TrimPrefix(name, pfx)
	}
	return strings.ReplaceAll(name, " ", "_")
}

func (p *GoogleProvider) parseToolCallsFromPayload(opener string, raw string, availableTools []Tool) []ToolCall {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}

	var result []ToolCall

	if opener == "[Called Tool:" {
		lower := strings.ToLower(raw)
		var funcName, argsStr string
		if idx := strings.Index(lower, "with arguments:"); idx != -1 {
			funcName = strings.TrimSpace(raw[:idx])
			argsStr = strings.TrimSpace(raw[idx+len("with arguments:"):])
		} else if idx := strings.Index(raw, "("); idx != -1 && strings.HasSuffix(raw, ")") {
			funcName = strings.TrimSpace(raw[:idx])
			argsStr = strings.TrimSpace(raw[idx+1 : len(raw)-1])
		} else if idx := strings.Index(raw, "{"); idx != -1 {
			funcName = strings.TrimSpace(raw[:idx])
			argsStr = strings.TrimSpace(raw[idx:])
		} else {
			funcName = raw
			argsStr = "{}"
		}

		funcName = cleanFunctionName(funcName)
		if funcName == "" {
			return nil
		}

		var argsMap map[string]any
		if err := json.Unmarshal([]byte(argsStr), &argsMap); err != nil {
			cleaned := cleanJSONString(argsStr)
			_ = json.Unmarshal([]byte(cleaned), &argsMap)
		}
		if argsMap == nil {
			argsMap = make(map[string]any)
		}
		fixedArgs, _ := p.fixToolCall(funcName, argsMap, availableTools)
		fixedBytes, _ := json.Marshal(fixedArgs)
		result = append(result, ToolCall{
			ID:   fmt.Sprintf("call_ws_text_%d", time.Now().UnixNano()),
			Type: "function",
			Function: FunctionCall{
				Name:      funcName,
				Arguments: string(fixedBytes),
			},
		})
		return result
	}

	if opener == "call:" {
		re := regexp.MustCompile(`^([a-zA-Z0-9_\-]+)\s*(\{.*)`)
		if m := re.FindStringSubmatch(raw); len(m) >= 3 {
			funcName := cleanFunctionName(m[1])
			argsStr := m[2]
			var argsMap map[string]any
			if err := json.Unmarshal([]byte(argsStr), &argsMap); err != nil {
				cleaned := cleanJSONString(argsStr)
				_ = json.Unmarshal([]byte(cleaned), &argsMap)
			}
			if argsMap == nil {
				argsMap = make(map[string]any)
			}
			fixedArgs, _ := p.fixToolCall(funcName, argsMap, availableTools)
			fixedBytes, _ := json.Marshal(fixedArgs)
			result = append(result, ToolCall{
				ID:   fmt.Sprintf("call_ws_text_%d", time.Now().UnixNano()),
				Type: "function",
				Function: FunctionCall{
					Name:      funcName,
					Arguments: string(fixedBytes),
				},
			})
		}
		return result
	}

	var items []map[string]any
	if err := json.Unmarshal([]byte(raw), &items); err != nil {
		var single map[string]any
		if err2 := json.Unmarshal([]byte(raw), &single); err2 == nil {
			items = append(items, single)
		} else {
			cleaned := cleanJSONString(raw)
			if err3 := json.Unmarshal([]byte(cleaned), &items); err3 != nil {
				if err4 := json.Unmarshal([]byte(cleaned), &single); err4 == nil {
					items = append(items, single)
				}
			}
		}
	}

	for i, item := range items {
		funcName := ""
		if n, ok := item["name"].(string); ok {
			funcName = n
		} else if t, ok := item["tool"].(string); ok {
			funcName = t
		} else if f, ok := item["function"].(string); ok {
			funcName = f
		}
		funcName = cleanFunctionName(funcName)
		if funcName == "" {
			continue
		}

		var args map[string]any
		if a, ok := item["arguments"].(map[string]any); ok {
			args = a
		} else if a, ok := item["args"].(map[string]any); ok {
			args = a
		} else if a, ok := item["parameters"].(map[string]any); ok {
			args = a
		} else if aStr, ok := item["arguments"].(string); ok {
			_ = json.Unmarshal([]byte(aStr), &args)
		} else {
			args = make(map[string]any)
			for k, v := range item {
				if k != "name" && k != "tool" && k != "function" {
					args[k] = v
				}
			}
		}
		if args == nil {
			args = make(map[string]any)
		}
		fixedArgs, _ := p.fixToolCall(funcName, args, availableTools)
		fixedBytes, _ := json.Marshal(fixedArgs)
		result = append(result, ToolCall{
			ID:   fmt.Sprintf("call_ws_text_%d_%d", time.Now().UnixNano(), i),
			Type: "function",
			Function: FunctionCall{
				Name:      funcName,
				Arguments: string(fixedBytes),
			},
		})
	}

	return result
}

func (p *GoogleProvider) parseUnclosedToolCalls(text string, availableTools []Tool) []ToolCall {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	patterns := []string{
		"<tool_calls>",
		"<tool_call>",
		"<tool_code>",
		"<toolUse>",
		"<tool_use>",
		"<function_calls>",
		"[Called Tool:",
	}

	for _, opener := range patterns {
		if idx := strings.Index(text, opener); idx != -1 {
			payload := text[idx+len(opener):]
			closer := ""
			for _, pat := range knownToolPatterns {
				if pat.opener == opener {
					closer = pat.closer
					break
				}
			}
			if closer != "" {
				payload = strings.TrimSuffix(strings.TrimSpace(payload), closer)
			}
			tcs := p.parseToolCallsFromPayload(opener, payload, availableTools)
			if len(tcs) > 0 {
				return tcs
			}
		}
	}
	return nil
}

func (p *GoogleProvider) processTextStream(
	pendingText string,
	incomingText string,
	availableTools []Tool,
	onToolCall func(ToolCall),
	onContent func(string),
) string {
	combined := pendingText + incomingText

	for len(combined) > 0 {
		bestOpenerIdx := -1
		var matchedPattern toolTagPattern

		for _, pat := range knownToolPatterns {
			idx := strings.Index(combined, pat.opener)
			if idx != -1 && (bestOpenerIdx == -1 || idx < bestOpenerIdx) {
				bestOpenerIdx = idx
				matchedPattern = pat
			}
		}

		if bestOpenerIdx == -1 {
			partialLen := 0
			for _, pat := range knownToolPatterns {
				for l := len(pat.opener) - 1; l >= 2; l-- {
					if strings.HasSuffix(combined, pat.opener[:l]) {
						if l > partialLen {
							partialLen = l
						}
					}
				}
			}
			if partialLen > 0 {
				safeToYield := combined[:len(combined)-partialLen]
				if safeToYield != "" && onContent != nil {
					onContent(safeToYield)
				}
				return combined[len(combined)-partialLen:]
			}
			if combined != "" && onContent != nil {
				onContent(combined)
			}
			return ""
		}

		if bestOpenerIdx > 0 {
			pre := combined[:bestOpenerIdx]
			if onContent != nil {
				onContent(pre)
			}
		}

		searchStart := bestOpenerIdx + len(matchedPattern.opener)
		closerIdx := strings.Index(combined[searchStart:], matchedPattern.closer)
		if closerIdx == -1 {
			return combined[bestOpenerIdx:]
		}

		rawPayload := combined[searchStart : searchStart+closerIdx]
		blockEnd := searchStart + closerIdx + len(matchedPattern.closer)

		tcs := p.parseToolCallsFromPayload(matchedPattern.opener, rawPayload, availableTools)
		for _, tc := range tcs {
			if onToolCall != nil {
				onToolCall(tc)
			}
		}

		combined = combined[blockEnd:]
	}

	return ""
}

func (p *GoogleProvider) flushTextStream(
	pendingText string,
	availableTools []Tool,
	onToolCall func(ToolCall),
	onContent func(string),
) {
	if pendingText == "" {
		return
	}
	tcs := p.parseUnclosedToolCalls(pendingText, availableTools)
	if len(tcs) > 0 {
		for _, tc := range tcs {
			if onToolCall != nil {
				onToolCall(tc)
			}
		}
		return
	}
	if onContent != nil {
		onContent(pendingText)
	}
}

func sanitizeGeminiSchemaMap(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	res := sanitizeGeminiSchema(m)
	if resultMap, ok := res.(map[string]any); ok {
		return resultMap
	}
	return m
}

func sanitizeGeminiSchema(val any) any {
	switch v := val.(type) {
	case map[string]any:
		cleaned := make(map[string]any)
		for k, child := range v {
			switch k {
			case "$comment", "$schema", "$id", "title", "enumDescriptions", "examples", "default", "additionalProperties",
				"pattern", "format", "minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems", "uniqueItems":
				continue
			}
			cleaned[k] = sanitizeGeminiSchema(child)
		}

		if typeVal, exists := cleaned["type"]; exists {
			if typeStr, ok := typeVal.(string); ok {
				cleaned["type"] = strings.ToUpper(typeStr)
			} else if typeSlice, ok := typeVal.([]any); ok {
				firstType := ""
				isNullable := false
				for _, t := range typeSlice {
					if s, ok := t.(string); ok {
						if s == "null" {
							isNullable = true
						} else if firstType == "" {
							firstType = s
						}
					}
				}
				if firstType != "" {
					cleaned["type"] = strings.ToUpper(firstType)
				} else {
					cleaned["type"] = "STRING"
				}
				if isNullable {
					cleaned["nullable"] = true
				}
			}
		}

		if t, ok := cleaned["type"].(string); ok && t == "OBJECT" {
			if cleaned["properties"] == nil {
				cleaned["properties"] = map[string]any{}
			}
		}

		if t, ok := cleaned["type"].(string); ok && t == "ARRAY" {
			if cleaned["items"] == nil {
				cleaned["items"] = map[string]any{"type": "STRING"}
			}
		}

		if reqVal, hasReq := cleaned["required"]; hasReq {
			propsMap, _ := cleaned["properties"].(map[string]any)
			var reqSlice []string
			switch r := reqVal.(type) {
			case []any:
				for _, item := range r {
					if s, ok := item.(string); ok {
						reqSlice = append(reqSlice, s)
					}
				}
			case []string:
				reqSlice = r
			}

			if propsMap == nil || len(propsMap) == 0 {
				delete(cleaned, "required")
			} else {
				validReq := make([]any, 0, len(reqSlice))
				for _, reqKey := range reqSlice {
					if _, exists := propsMap[reqKey]; exists {
						validReq = append(validReq, reqKey)
					}
				}
				if len(validReq) > 0 {
					cleaned["required"] = validReq
				} else {
					delete(cleaned, "required")
				}
			}
		}

		return cleaned

	case []any:
		cleanedSlice := make([]any, len(v))
		for i, item := range v {
			cleanedSlice[i] = sanitizeGeminiSchema(item)
		}
		return cleanedSlice

	default:
		return val
	}
}

// --- Google WebSocket Live Implementation ---

type liveSetupMessage struct {
	Setup liveSetup `json:"setup"`
}

type liveSetup struct {
	Model                    string                `json:"model"`
	GenerationConfig         *liveGenerationConfig `json:"generationConfig,omitempty"`
	SystemInstruction        *geminiContent        `json:"systemInstruction,omitempty"`
	OutputAudioTranscription map[string]any        `json:"outputAudioTranscription,omitempty"`
	Tools                    []liveTool            `json:"tools,omitempty"`
	SafetySettings           []geminiSafetySetting `json:"safetySettings,omitempty"`
}

type liveGenerationConfig struct {
	ResponseModalities []string      `json:"responseModalities,omitempty"`
	ThinkingConfig     *liveThinking `json:"thinkingConfig,omitempty"`
}

type liveThinking struct {
	IncludeThoughts bool   `json:"includeThoughts"`
	ThinkingLevel   string `json:"thinkingLevel,omitempty"`
	ThinkingBudget  *int   `json:"thinkingBudget,omitempty"`
}

type liveTool struct {
	FunctionDeclarations []geminiFunction `json:"functionDeclarations,omitempty"`
}

type liveClientContentMessage struct {
	ClientContent liveClientContent `json:"clientContent"`
}

type liveClientContent struct {
	Turns        []geminiContent `json:"turns"`
	TurnComplete bool            `json:"turnComplete"`
}

type liveIncomingFrame struct {
	SetupComplete any `json:"setupComplete,omitempty"`
	ServerContent *struct {
		ModelTurn *struct {
			Parts []struct {
				Text             string              `json:"text,omitempty"`
				Thought          bool                `json:"thought,omitempty"`
				ThoughtSignature string              `json:"thoughtSignature,omitempty"`
				InlineData       *geminiInline       `json:"inlineData,omitempty"`
				FunctionCall     *geminiFunctionCall `json:"functionCall,omitempty"`
			} `json:"parts,omitempty"`
		} `json:"modelTurn,omitempty"`
		OutputTranscription *struct {
			Text string `json:"text,omitempty"`
		} `json:"outputTranscription,omitempty"`
		TurnComplete       bool   `json:"turnComplete,omitempty"`
		GenerationComplete bool   `json:"generationComplete,omitempty"`
		Interrupted        bool   `json:"interrupted,omitempty"`
		InteractionStatus  string `json:"interactionStatus,omitempty"`
	} `json:"serverContent,omitempty"`
	ToolCall *struct {
		FunctionCalls []struct {
			ID   string         `json:"id,omitempty"`
			Name string         `json:"name,omitempty"`
			Args map[string]any `json:"args,omitempty"`
		} `json:"functionCalls,omitempty"`
	} `json:"toolCall,omitempty"`
}

type liveConnSession struct {
	conn      *websocket.Conn
	expiresAt time.Time
}

var (
	liveConnsLock sync.Mutex
	liveConns     = make(map[string]*liveConnSession)
)

func cleanExpiredLiveConns() {
	liveConnsLock.Lock()
	defer liveConnsLock.Unlock()
	now := time.Now()
	for id, sess := range liveConns {
		if now.After(sess.expiresAt) {
			if sess.conn != nil {
				sess.conn.Close()
			}
			delete(liveConns, id)
		}
	}
}

func (p *GoogleProvider) dialAndSetupLive(ctx context.Context, apiKey string, targetModel string, req *CompletionRequest) (*websocket.Conn, []geminiContent, error) {
	model := targetModel
	if model == "" {
		model = "gemini-3.1-flash-live-preview"
	}

	modelLower := strings.ToLower(model)
	var thinkingLevel string
	if strings.Contains(modelLower, "extended-thinking") || strings.Contains(modelLower, "high") {
		thinkingLevel = "HIGH"
	} else if strings.Contains(modelLower, "low") {
		thinkingLevel = "LOW"
	} else if strings.Contains(modelLower, "minimal") {
		thinkingLevel = "MINIMAL"
	} else if strings.Contains(modelLower, "medium") {
		thinkingLevel = "MEDIUM"
	} else if strings.Contains(modelLower, "3.8") {
		thinkingLevel = "LOW"
	} else {
		thinkingLevel = "HIGH"
	}

	if req.Thinking != nil && req.Thinking.ThinkingLevel != "" {
		thinkingLevel = strings.ToUpper(string(req.Thinking.ThinkingLevel))
	}

	if !strings.HasPrefix(model, "models/") {
		model = "models/" + model
	}

	wsURL := p.WSURL
	if wsURL == "" {
		wsURL = fmt.Sprintf("wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key=%s", url.QueryEscape(apiKey))
	}

	dialer := websocket.DefaultDialer
	conn, resp, err := dialer.DialContext(ctx, wsURL, nil)
	if err != nil {
		if resp != nil {
			return nil, nil, fmt.Errorf("google live ws dial failed with status %d: %w", resp.StatusCode, err)
		}
		return nil, nil, fmt.Errorf("google live ws dial failed: %w", err)
	}

	var systemInstructions []string
	contents, systemInstructions := p.buildGeminiContents(req.Messages, true)
	if req.SystemInstruction != "" {
		systemInstructions = append([]string{req.SystemInstruction}, systemInstructions...)
	}

	modalities := p.ResponseModalities
	if len(modalities) == 0 {
		modalities = []string{"AUDIO"}
	}

	setupMsg := liveSetupMessage{
		Setup: liveSetup{
			Model: model,
			GenerationConfig: &liveGenerationConfig{
				ResponseModalities: modalities,
				ThinkingConfig: &liveThinking{
					IncludeThoughts: true,
					ThinkingLevel:   thinkingLevel,
				},
			},
			OutputAudioTranscription: map[string]any{},
			SafetySettings: []geminiSafetySetting{
				{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_NONE"},
				{Category: "HARM_CATEGORY_HATE_SPEECH", Threshold: "BLOCK_NONE"},
				{Category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", Threshold: "BLOCK_NONE"},
				{Category: "HARM_CATEGORY_DANGEROUS_CONTENT", Threshold: "BLOCK_NONE"},
				{Category: "HARM_CATEGORY_CIVIC_INTEGRITY", Threshold: "BLOCK_NONE"},
			},
		},
	}

	if len(systemInstructions) > 0 {
		setupMsg.Setup.SystemInstruction = &geminiContent{
			Parts: []geminiPart{{Text: strings.Join(systemInstructions, "\n\n")}},
		}
	}

	if len(req.Tools) > 0 {
		var funcDecls []geminiFunction
		for _, tool := range req.Tools {
			for _, fn := range tool.Functions {
				funcDecls = append(funcDecls, geminiFunction{
					Name:        fn.Name,
					Description: fn.Description,
					Parameters:  sanitizeGeminiSchemaMap(fn.Parameters),
				})
			}
			if tool.Name != "" && len(tool.Functions) == 0 {
				funcDecls = append(funcDecls, geminiFunction{
					Name:        tool.Name,
					Description: tool.Description,
					Parameters:  sanitizeGeminiSchemaMap(tool.Parameters),
				})
			}
		}
		if len(funcDecls) > 0 {
			setupMsg.Setup.Tools = []liveTool{{FunctionDeclarations: funcDecls}}
		}
	}

	setupJSON, err := json.Marshal(setupMsg)
	if err != nil {
		conn.Close()
		return nil, nil, fmt.Errorf("failed to marshal setup frame: %w", err)
	}

	if err := conn.WriteMessage(websocket.TextMessage, setupJSON); err != nil {
		conn.Close()
		return nil, nil, fmt.Errorf("failed to send setup frame: %w", err)
	}

	return conn, contents, nil
}

func (p *GoogleProvider) readLiveStream(ctx context.Context, conn *websocket.Conn, contents []geminiContent, req *CompletionRequest, onChunk func(*CompletionResponse)) (*CompletionResponse, bool, error) {
	var (
		fullResponse CompletionResponse
		keepConnOpen bool
		doneChan     = make(chan struct{})
		errChan      = make(chan error, 1)
	)

	go func() {
		select {
		case <-ctx.Done():
			if !keepConnOpen && conn != nil {
				conn.Close()
			}
		case <-doneChan:
		}
	}()

	go func() {
		defer close(doneChan)
		var pendingText string
		for {
			_, msgBytes, err := conn.ReadMessage()
			if err != nil {
				if websocket.IsCloseError(err, websocket.CloseNormalClosure) {
					if fullResponse.Content == "" && len(fullResponse.ToolCalls) == 0 {
						errChan <- fmt.Errorf("gemini live ws closed with normal closure (1000) without response")
					}
					return
				}
				if ctx.Err() != nil {
					return
				}
				errChan <- err
				return
			}

			var frame liveIncomingFrame
			if err := json.Unmarshal(msgBytes, &frame); err != nil {
				continue
			}

			if frame.SetupComplete != nil {
				if len(contents) > 0 {
					clientContentMsg := liveClientContentMessage{
						ClientContent: liveClientContent{
							Turns:        contents,
							TurnComplete: true,
						},
					}
					clientContentJSON, _ := json.Marshal(clientContentMsg)
					if err := conn.WriteMessage(websocket.TextMessage, clientContentJSON); err != nil {
						errChan <- fmt.Errorf("failed to send clientContent frame: %w", err)
						return
					}
				}
				continue
			}

			if frame.ServerContent != nil {
				if frame.ServerContent.ModelTurn != nil {
					for _, part := range frame.ServerContent.ModelTurn.Parts {
						if part.Thought && part.Text != "" {
							fullResponse.Thought += part.Text
							if onChunk != nil {
								onChunk(&CompletionResponse{Thought: part.Text})
							}
						} else if part.FunctionCall != nil {
							funcName := part.FunctionCall.Name
							prefixes := []string{"tool_use:", "tool_code:", "tool_call:", "call:"}
							for _, pfx := range prefixes {
								funcName = strings.TrimPrefix(funcName, pfx)
							}
							rawArgsBytes, _ := json.Marshal(part.FunctionCall.Args)
							fixedArgs, _ := p.fixToolCall(funcName, part.FunctionCall.Args, req.Tools)
							argsBytes, _ := json.Marshal(fixedArgs)

							callID := part.FunctionCall.ID
							if callID == "" {
								callID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), len(fullResponse.ToolCalls))
							}
							cacheThoughtSignature(callID, funcName, part.ThoughtSignature)
							log.Printf("[DEBUG][WS -> PROXY] Live Tool Call (part) | Tool: %s, ID: %s, RawArgs: %s, FixedArgs: %s (sig len: %d)", funcName, callID, string(rawArgsBytes), string(argsBytes), len(part.ThoughtSignature))
							tc := ToolCall{
								ID:   callID,
								Type: "function",
								Function: FunctionCall{
									Name:      funcName,
									Arguments: string(argsBytes),
								},
								ThoughtSignature: part.ThoughtSignature,
							}
							fullResponse.ToolCalls = append(fullResponse.ToolCalls, tc)
							if onChunk != nil {
								onChunk(&CompletionResponse{ToolCalls: []ToolCall{tc}})
							}
						} else if part.Text != "" {
							pendingText = p.processTextStream(
								pendingText,
								part.Text,
								req.Tools,
								func(tc ToolCall) {
									log.Printf("[DEBUG][WS -> PROXY] Live Parsed Text Tool Call | Tool: %s, ID: %s, Args: %s", tc.Function.Name, tc.ID, tc.Function.Arguments)
									fullResponse.ToolCalls = append(fullResponse.ToolCalls, tc)
									if onChunk != nil {
										onChunk(&CompletionResponse{ToolCalls: []ToolCall{tc}})
									}
								},
								func(text string) {
									fullResponse.Content += text
									if onChunk != nil {
										onChunk(&CompletionResponse{Content: text})
									}
								},
							)
						}
					}
				}

				if frame.ServerContent.OutputTranscription != nil && frame.ServerContent.OutputTranscription.Text != "" {
					txt := frame.ServerContent.OutputTranscription.Text
					pendingText = p.processTextStream(
						pendingText,
						txt,
						req.Tools,
						func(tc ToolCall) {
							log.Printf("[DEBUG][WS -> PROXY] Live Parsed Transcription Tool Call | Tool: %s, ID: %s, Args: %s", tc.Function.Name, tc.ID, tc.Function.Arguments)
							fullResponse.ToolCalls = append(fullResponse.ToolCalls, tc)
							if onChunk != nil {
								onChunk(&CompletionResponse{ToolCalls: []ToolCall{tc}})
							}
						},
						func(text string) {
							fullResponse.Content += text
							if onChunk != nil {
								onChunk(&CompletionResponse{Content: text})
							}
						},
					)
				}

				if frame.ServerContent.TurnComplete || frame.ServerContent.GenerationComplete {
					p.flushTextStream(
						pendingText,
						req.Tools,
						func(tc ToolCall) {
							log.Printf("[DEBUG][WS -> PROXY] Live Flushed Text Tool Call | Tool: %s, ID: %s, Args: %s", tc.Function.Name, tc.ID, tc.Function.Arguments)
							fullResponse.ToolCalls = append(fullResponse.ToolCalls, tc)
							if onChunk != nil {
								onChunk(&CompletionResponse{ToolCalls: []ToolCall{tc}})
							}
						},
						func(text string) {
							fullResponse.Content += text
							if onChunk != nil {
								onChunk(&CompletionResponse{Content: text})
							}
						},
					)
					pendingText = ""

					if len(fullResponse.ToolCalls) > 0 {
						sess := &liveConnSession{
							conn:      conn,
							expiresAt: time.Now().Add(2 * time.Minute),
						}
						liveConnsLock.Lock()
						for _, tc := range fullResponse.ToolCalls {
							liveConns[tc.ID] = sess
						}
						liveConnsLock.Unlock()
						keepConnOpen = true
						return
					}

					// A bare generationComplete is NOT the end of the interaction when
					// tools are declared. The Live API emits generationComplete, then a
					// turnComplete carrying interactionStatus "IN_PROGRESS", and only
					// afterwards the toolCall frame (observed on
					// gemini-3.8-live-extended-thinking). Returning on the bare
					// generationComplete would drop the function call entirely.
					if frame.ServerContent.GenerationComplete && !frame.ServerContent.TurnComplete && len(req.Tools) > 0 {
						continue
					}
					if frame.ServerContent.InteractionStatus == "IN_PROGRESS" {
						continue
					}
					return
				}
			}

			if frame.ToolCall != nil && len(frame.ToolCall.FunctionCalls) > 0 {
				sess := &liveConnSession{
					conn:      conn,
					expiresAt: time.Now().Add(2 * time.Minute),
				}
				liveConnsLock.Lock()
				for _, fc := range frame.ToolCall.FunctionCalls {
					funcName := fc.Name
					prefixes := []string{"tool_use:", "tool_code:", "tool_call:", "call:"}
					for _, pfx := range prefixes {
						funcName = strings.TrimPrefix(funcName, pfx)
					}
					rawArgsBytes, _ := json.Marshal(fc.Args)
					fixedArgs, _ := p.fixToolCall(funcName, fc.Args, req.Tools)
					argsBytes, _ := json.Marshal(fixedArgs)

					callID := fc.ID
					if callID == "" {
						callID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), len(fullResponse.ToolCalls))
					}
					log.Printf("[DEBUG][WS -> PROXY] Live Tool Call (frame) | Tool: %s, ID: %s, RawArgs: %s, FixedArgs: %s", funcName, callID, string(rawArgsBytes), string(argsBytes))
					tc := ToolCall{
						ID:   callID,
						Type: "function",
						Function: FunctionCall{
							Name:      funcName,
							Arguments: string(argsBytes),
						},
					}
					fullResponse.ToolCalls = append(fullResponse.ToolCalls, tc)
					liveConns[callID] = sess
					if onChunk != nil {
						onChunk(&CompletionResponse{ToolCalls: []ToolCall{tc}})
					}
				}
				liveConnsLock.Unlock()
				keepConnOpen = true
				return
			}
		}
	}()

	select {
	case err := <-errChan:
		return nil, false, err
	case <-doneChan:
		return &fullResponse, keepConnOpen, nil
	case <-ctx.Done():
		return nil, false, ctx.Err()
	}
}

func (p *GoogleProvider) chatWebSocket(ctx context.Context, apiKey string, targetModel string, req *CompletionRequest, onChunk func(*CompletionResponse)) (*CompletionResponse, error) {
	cleanExpiredLiveConns()

	var matchedSession *liveConnSession
	for _, msg := range req.Messages {
		if (msg.Role == "tool" || msg.Role == "function") && msg.ToolCallID != "" {
			liveConnsLock.Lock()
			if sess, ok := liveConns[msg.ToolCallID]; ok {
				if time.Now().Before(sess.expiresAt) {
					matchedSession = sess
				}
			}
			liveConnsLock.Unlock()
			if matchedSession != nil {
				break
			}
		}
	}

	if matchedSession != nil {
		liveConnsLock.Lock()
		for id, sess := range liveConns {
			if sess == matchedSession {
				delete(liveConns, id)
			}
		}
		liveConnsLock.Unlock()

		var toolResponses []geminiFunctionResponse
		for i := len(req.Messages) - 1; i >= 0; i-- {
			msg := req.Messages[i]
			if msg.Role != "tool" && msg.Role != "function" {
				break
			}
			funcName := msg.Name
			if funcName == "" && msg.ToolCallID != "" {
				for j := i - 1; j >= 0; j-- {
					for _, tc := range req.Messages[j].ToolCalls {
						if tc.ID == msg.ToolCallID {
							funcName = tc.Function.Name
							break
						}
					}
					if funcName != "" {
						break
					}
				}
			}
			if funcName == "" {
				for j := i - 1; j >= 0; j-- {
					if len(req.Messages[j].ToolCalls) > 0 {
						funcName = req.Messages[j].ToolCalls[0].Function.Name
						break
					}
				}
			}
			for _, pfx := range []string{"tool_use:", "tool_code:", "tool_call:", "call:"} {
				funcName = strings.TrimPrefix(funcName, pfx)
			}
			if funcName == "" {
				funcName = "unknown"
			}

			toolText := ""
			for _, part := range msg.Content {
				if part.Type == ContentTypeText && part.Text != "" {
					toolText += part.Text
				}
			}

			respMap := map[string]any{"result": toolText}
			var jsonResult any
			if err := json.Unmarshal([]byte(toolText), &jsonResult); err == nil {
				if m, ok := jsonResult.(map[string]any); ok {
					respMap = m
				} else {
					respMap = map[string]any{"result": jsonResult}
				}
			}

			toolResponses = append([]geminiFunctionResponse{
				{
					Name:     funcName,
					Response: respMap,
					ID:       msg.ToolCallID,
				},
			}, toolResponses...)
		}

		if len(toolResponses) > 0 {
			toolRespMsg := map[string]any{
				"toolResponse": map[string]any{
					"functionResponses": toolResponses,
				},
			}
			toolRespJSON, err := json.Marshal(toolRespMsg)
			if err == nil {
				log.Printf("[DEBUG][WS -> GEMINI] Reusing connection for %d tool response(s)...", len(toolResponses))
				if err := matchedSession.conn.WriteMessage(websocket.TextMessage, toolRespJSON); err == nil {
					chunksYielded := 0
					safeOnChunk := func(c *CompletionResponse) {
						if c != nil && (c.Content != "" || len(c.ToolCalls) > 0 || c.Thought != "") {
							chunksYielded++
							if onChunk != nil {
								onChunk(c)
							}
						}
					}

					resp, keepOpen, readErr := p.readLiveStream(ctx, matchedSession.conn, nil, req, safeOnChunk)
					if !keepOpen {
						matchedSession.conn.Close()
					}

					if readErr == nil && resp != nil && (resp.Content != "" || len(resp.ToolCalls) > 0) {
						return resp, nil
					}

					if chunksYielded == 0 && ctx.Err() == nil {
						log.Printf("[WARN][WS] Reused connection failed or closed without response (%v). Discarding session and falling back to fresh connection...", readErr)
					} else {
						return resp, readErr
					}
				} else {
					log.Printf("[WARN][WS] Failed to write toolResponse to cached connection (%v), falling back to fresh connection", err)
					matchedSession.conn.Close()
				}
			}
		} else {
			matchedSession.conn.Close()
		}
	}

	// Dial fresh connection
	conn, contents, err := p.dialAndSetupLive(ctx, apiKey, targetModel, req)
	if err != nil {
		return nil, err
	}
	var keepConnOpen bool
	defer func() {
		if !keepConnOpen && conn != nil {
			conn.Close()
		}
	}()

	resp, keepOpen, err := p.readLiveStream(ctx, conn, contents, req, onChunk)
	keepConnOpen = keepOpen
	return resp, err
}
