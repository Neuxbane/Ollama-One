package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/Neuxbane/Ollama-One/providers"
)

type ConfigEntry struct {
	Type     string `json:"type"`
	Provider string `json:"provider"`
	Key      string `json:"key"`
}

func (c ConfigEntry) GetType() string {
	if c.Type != "" {
		return c.Type
	}
	return c.Provider
}

type loggingResponseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (lrw *loggingResponseWriter) WriteHeader(code int) {
	lrw.statusCode = code
	lrw.ResponseWriter.WriteHeader(code)
}

type OpenAIToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type OpenAITool struct {
	Type     string             `json:"type"`
	Function OpenAIToolFunction `json:"function"`
}

type OpenAIToolCallFunction struct {
	Name      string `json:"name"`
	Arguments any    `json:"arguments"`
}

type OpenAIToolCall struct {
	Index    *int                   `json:"index,omitempty"`
	ID       string                 `json:"id,omitempty"`
	Type     string                 `json:"type,omitempty"`
	Function OpenAIToolCallFunction `json:"function"`
}

type OpenAIChatMessage struct {
	Role       string           `json:"role"`
	Name       string           `json:"name,omitempty"`
	Content    any              `json:"content"`
	ToolCalls  []OpenAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	Images     []string         `json:"images,omitempty"`
}

type OpenAIStreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

type OpenAIChatCompletionRequest struct {
	Model               string               `json:"model"`
	Messages            []OpenAIChatMessage  `json:"messages"`
	Stream              bool                 `json:"stream"`
	StreamOptions       *OpenAIStreamOptions `json:"stream_options,omitempty"`
	Tools               []OpenAITool         `json:"tools,omitempty"`
	ToolChoice          any                  `json:"tool_choice,omitempty"`
	Functions           []OpenAIToolFunction `json:"functions,omitempty"`
	FunctionCall        any                  `json:"function_call,omitempty"`
	Temperature         *float64             `json:"temperature,omitempty"`
	TopP                *float64             `json:"top_p,omitempty"`
	MaxTokens           *int                 `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int                 `json:"max_completion_tokens,omitempty"`
	ReasoningEffort     string               `json:"reasoning_effort,omitempty"`
	SessionID           string               `json:"session_id,omitempty"`
}

var (
	googleProvider providers.Provider
	sessionManager = NewSessionManager()
)

func loadConfig(path string) ([]ConfigEntry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var config []ConfigEntry
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}
	return config, nil
}

func main() {
	config, err := loadConfig("config.json")
	if err != nil {
		log.Printf("Warning: could not load config.json: %v", err)
	}

	var googleKeys []string

	for _, entry := range config {
		t := strings.ToLower(entry.GetType())
		switch t {
		case "google", "gemini", "gemini-live", "live":
			if entry.Key != "" {
				googleKeys = append(googleKeys, entry.Key)
			}
		default:
			log.Printf("[CONFIG] Warning: ignored provider type %q in config", entry.GetType())
		}
	}

	// Fallback to environment variables
	if len(googleKeys) == 0 {
		if envKey := os.Getenv("GEMINI_API_KEY"); envKey != "" {
			googleKeys = append(googleKeys, envKey)
		} else if envKey := os.Getenv("GOOGLE_API_KEY"); envKey != "" {
			googleKeys = append(googleKeys, envKey)
		}
	}

	log.Printf("[CONFIG] Loaded %d Google API key(s)", len(googleKeys))

	googleProvider = providers.NewGoogleProvider(googleKeys)

	mux := http.NewServeMux()

	// OpenAI-compatible chat completions
	mux.HandleFunc("/v1/chat/completions", handleChatCompletions)
	mux.HandleFunc("/chat/completions", handleChatCompletions)

	// OpenAI-compatible models endpoints
	mux.HandleFunc("/v1/models", handleModels)
	mux.HandleFunc("/models", handleModels)
	mux.HandleFunc("/v1/models/", handleModels)
	mux.HandleFunc("/models/", handleModels)

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		json.NewEncoder(w).Encode(map[string]any{
			"status":  "ok",
			"service": "google-ai-proxy",
		})
	})

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		reqHeaders := r.Header.Get("Access-Control-Request-Headers")
		if reqHeaders != "" {
			w.Header().Set("Access-Control-Allow-Headers", reqHeaders)
		} else {
			w.Header().Set("Access-Control-Allow-Headers", "*")
		}

		lrw := &loggingResponseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		if r.Method == "OPTIONS" {
			lrw.WriteHeader(http.StatusNoContent)
			log.Printf("[HTTP] OPTIONS %s -> 204 (%v)", r.URL.Path, time.Since(start))
			return
		}
		if r.URL.Path == "/favicon.ico" {
			lrw.WriteHeader(http.StatusNotFound)
			return
		}

		log.Printf("[HTTP] -> %s %s (from %s)", r.Method, r.URL.RequestURI(), r.RemoteAddr)
		mux.ServeHTTP(lrw, r)
		log.Printf("[HTTP] <- %s %s finished -> %d (%v)", r.Method, r.URL.RequestURI(), lrw.statusCode, time.Since(start))
	})

	portEnv := os.Getenv("PORT")
	if portEnv == "" {
		portEnv = os.Getenv("OLLAMA_PORT")
	}

	host := os.Getenv("HOST")
	if host == "" {
		host = "0.0.0.0"
	}

	portsToTry := []string{"11434", "11435"}
	if portEnv != "" {
		portsToTry = []string{portEnv}
	}

	for _, p := range portsToTry {
		addr := fmt.Sprintf("%s:%s", host, p)
		listener, err := net.Listen("tcp", addr)
		if err == nil {
			fmt.Printf("Google AI Proxy starting on http://%s...\n", addr)
			if err := http.Serve(listener, handler); err != nil {
				fmt.Printf("Error running server: %v\n", err)
			}
			return
		}
		fmt.Printf("Port %s unavailable (%v), trying next port...\n", p, err)
	}
}

func extractOpenAITextContent(content any) string {
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var b strings.Builder
		for _, item := range v {
			switch p := item.(type) {
			case string:
				b.WriteString(p)
			case map[string]any:
				if p["type"] == "text" {
					if text, ok := p["text"].(string); ok {
						b.WriteString(text)
					}
				}
			}
		}
		return b.String()
	default:
		return ""
	}
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req OpenAIChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	if req.Model == "" {
		req.Model = "gemini-2.0-flash"
	}
	req.Model = strings.TrimSpace(strings.TrimSuffix(req.Model, ":latest"))

	// Convert Tools
	var internalTools []providers.Tool
	for _, tool := range req.Tools {
		name := tool.Function.Name
		desc := tool.Function.Description
		params := tool.Function.Parameters
		if params == nil {
			params = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		tType := tool.Type
		if tType == "" {
			tType = "function"
		}
		internalTools = append(internalTools, providers.Tool{
			Type:        tType,
			Name:        name,
			Description: desc,
			Parameters:  params,
			Functions: []providers.Tool{
				{
					Type:        "function",
					Name:        name,
					Description: desc,
					Parameters:  params,
				},
			},
		})
	}
	if len(internalTools) == 0 && len(req.Functions) > 0 {
		for _, f := range req.Functions {
			params := f.Parameters
			if params == nil {
				params = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			internalTools = append(internalTools, providers.Tool{
				Type:        "function",
				Name:        f.Name,
				Description: f.Description,
				Parameters:  params,
				Functions: []providers.Tool{
					{
						Type:        "function",
						Name:        f.Name,
						Description: f.Description,
						Parameters:  params,
					},
				},
			})
		}
	}

	// Convert Messages
	internalMessages := make([]providers.Message, len(req.Messages))
	for i, m := range req.Messages {
		parts := parseOpenAIMessageParts(m.Content, m.Images)

		var tcs []providers.ToolCall
		for _, tc := range m.ToolCalls {
			argsStr := ""
			switch a := tc.Function.Arguments.(type) {
			case string:
				argsStr = a
			case map[string]any:
				b, _ := json.Marshal(a)
				argsStr = string(b)
			default:
				if a != nil {
					b, _ := json.Marshal(a)
					argsStr = string(b)
				} else {
					argsStr = "{}"
				}
			}
			tcID := tc.ID
			if tcID == "" {
				tcID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), len(tcs))
			}
			tcType := tc.Type
			if tcType == "" {
				tcType = "function"
			}
			tcs = append(tcs, providers.ToolCall{
				ID:   tcID,
				Type: tcType,
				Function: providers.FunctionCall{
					Name:      tc.Function.Name,
					Arguments: argsStr,
				},
			})
		}

		internalMessages[i] = providers.Message{
			Role:       m.Role,
			Name:       m.Name,
			ToolCallID: m.ToolCallID,
			Content:    parts,
			ToolCalls:  tcs,
		}

		logText := extractOpenAITextContent(m.Content)
		if len(parts) > 0 && parts[0].Text != "" {
			logText = parts[0].Text
		}

		displayLogText := logText
		if m.Role == "system" || m.Role == "developer" {
			firstLine := strings.SplitN(strings.TrimSpace(logText), "\n", 2)[0]
			if len(firstLine) > 100 {
				firstLine = firstLine[:100] + "..."
			}
			displayLogText = fmt.Sprintf("%s (%d chars)", firstLine, len(logText))
		} else if len(displayLogText) > 400 {
			displayLogText = displayLogText[:400] + fmt.Sprintf("... [truncated %d chars]", len(logText)-400)
		}

		log.Printf("Client -> Proxy: [%s] %s (parts: %d, tool_calls: %d)", m.Role, displayLogText, len(parts), len(tcs))

		if m.Role == "tool" {
			log.Printf("[DEBUG][CLIENT -> PROXY] Tool Result | CallID: %s, Name: %s, Result: %s", m.ToolCallID, m.Name, displayLogText)
		}
		for _, tc := range tcs {
			log.Printf("[DEBUG][CLIENT -> PROXY] Past Tool Call | ID: %s, Function: %s, Arguments: %s", tc.ID, tc.Function.Name, tc.Function.Arguments)
		}
	}

	internalReq := &providers.CompletionRequest{
		Model:    req.Model,
		Messages: internalMessages,
		Tools:    internalTools,
		Stream:   req.Stream,
	}

	if req.ReasoningEffort != "" {
		internalReq.Thinking = &providers.ThinkingConfig{
			IncludeThoughts: true,
			ThinkingLevel:   providers.ThinkingLevel(strings.ToLower(req.ReasoningEffort)),
		}
	}

	// Session management
	var session *Session
	if req.SessionID != "" {
		session = sessionManager.GetSession(req.SessionID)
		if len(internalReq.Messages) > 0 {
			lastClientMsg := internalReq.Messages[len(internalReq.Messages)-1]
			if len(session.Messages) == 0 {
				session.Messages = internalReq.Messages
			} else {
				found := false
				for _, sm := range session.Messages {
					if sm.Role == lastClientMsg.Role && len(sm.Content) > 0 && len(lastClientMsg.Content) > 0 && sm.Content[0].Text == lastClientMsg.Content[0].Text {
						found = true
						break
					}
				}
				if !found {
					session.Messages = append(session.Messages, lastClientMsg)
				}
			}
			internalReq.Messages = session.Messages
		}
	}

	completionID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	createdTime := time.Now().Unix()

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("Transfer-Encoding", "chunked")
		w.Header().Set("X-Accel-Buffering", "no")

		flusher, ok := w.(http.Flusher)
		hasToolCalls := false

		// Emit initial role chunk immediately for VS Code Copilot
		initChunk := map[string]any{
			"id":                 completionID,
			"object":             "chat.completion.chunk",
			"created":            createdTime,
			"model":              req.Model,
			"system_fingerprint": "fp_google_proxy",
			"choices": []any{
				map[string]any{
					"index":         0,
					"delta":         map[string]any{"role": "assistant"},
					"finish_reason": nil,
				},
			},
		}
		initData, _ := json.Marshal(initChunk)
		fmt.Fprintf(w, "data: %s\n\n", initData)
		if ok {
			flusher.Flush()
		}

		var fullContent strings.Builder
		var allStreamToolCalls []providers.ToolCall

		_, err := googleProvider.Chat(r.Context(), internalReq, func(chunk *providers.CompletionResponse) {
			delta := map[string]any{}

			if chunk.Content != "" {
				delta["content"] = chunk.Content
				fullContent.WriteString(chunk.Content)
			}

			if len(chunk.ToolCalls) > 0 {
				hasToolCalls = true
				allStreamToolCalls = append(allStreamToolCalls, chunk.ToolCalls...)
				var openaiTCs []map[string]any
				for i, tc := range chunk.ToolCalls {
					tcID := tc.ID
					if tcID == "" {
						tcID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), i)
					}
					tcArgs := tc.Function.Arguments
					if tcArgs == "" {
						tcArgs = "{}"
					}
					log.Printf("[DEBUG][PROXY -> CLIENT] Stream Tool Call | Index: %d, ID: %s, Function: %s, Arguments: %s", i, tcID, tc.Function.Name, tcArgs)
					openaiTCs = append(openaiTCs, map[string]any{
						"index": i,
						"id":    tcID,
						"type":  "function",
						"function": map[string]any{
							"name":      tc.Function.Name,
							"arguments": tcArgs,
						},
					})
				}
				delta["tool_calls"] = openaiTCs
			}

			if len(delta) == 0 {
				return
			}

			chunkResp := map[string]any{
				"id":                 completionID,
				"object":             "chat.completion.chunk",
				"created":            createdTime,
				"model":              req.Model,
				"system_fingerprint": "fp_google_proxy",
				"choices": []any{
					map[string]any{
						"index":         0,
						"delta":         delta,
						"finish_reason": nil,
					},
				},
			}

			data, _ := json.Marshal(chunkResp)
			fmt.Fprintf(w, "data: %s\n\n", data)
			if ok {
				flusher.Flush()
			}
		})

		if err != nil {
			log.Printf("[Stream] Chat error: %v", err)
			errChunk := map[string]any{
				"id":                 completionID,
				"object":             "chat.completion.chunk",
				"created":            createdTime,
				"model":              req.Model,
				"system_fingerprint": "fp_google_proxy",
				"choices": []any{
					map[string]any{
						"index":         0,
						"delta":         map[string]any{"content": fmt.Sprintf("\n[Error: %v]", err)},
						"finish_reason": "stop",
					},
				},
			}
			errData, _ := json.Marshal(errChunk)
			fmt.Fprintf(w, "data: %s\n\n", errData)
			fmt.Fprint(w, "data: [DONE]\n\n")
			if ok {
				flusher.Flush()
			}
			return
		}

		if session != nil {
			session.Messages = append(session.Messages, providers.Message{
				Role:      "assistant",
				Content:   []providers.ContentPart{{Type: providers.ContentTypeText, Text: fullContent.String()}},
				ToolCalls: allStreamToolCalls,
			})
		}

		finishReason := "stop"
		if hasToolCalls {
			finishReason = "tool_calls"
		}

		endResp := map[string]any{
			"id":                 completionID,
			"object":             "chat.completion.chunk",
			"created":            createdTime,
			"model":              req.Model,
			"system_fingerprint": "fp_google_proxy",
			"choices": []any{
				map[string]any{
					"index":         0,
					"delta":         map[string]any{},
					"finish_reason": finishReason,
				},
			},
		}

		endData, _ := json.Marshal(endResp)
		fmt.Fprintf(w, "data: %s\n\n", endData)

		if req.StreamOptions != nil && req.StreamOptions.IncludeUsage {
			promptTokens := estimateTokens(internalMessages)
			completionTokens := estimateStringTokens(fullContent.String())
			usageResp := map[string]any{
				"id":                 completionID,
				"object":             "chat.completion.chunk",
				"created":            createdTime,
				"model":              req.Model,
				"system_fingerprint": "fp_google_proxy",
				"choices": []any{
					map[string]any{
						"index":         0,
						"delta":         map[string]any{},
						"finish_reason": nil,
					},
				},
				"usage": map[string]any{
					"prompt_tokens":     promptTokens,
					"completion_tokens": completionTokens,
					"total_tokens":      promptTokens + completionTokens,
				},
			}
			usageData, _ := json.Marshal(usageResp)
			fmt.Fprintf(w, "data: %s\n\n", usageData)
		}

		fmt.Fprint(w, "data: [DONE]\n\n")
		if ok {
			flusher.Flush()
		}
		return
	}

	resp, err := googleProvider.Chat(r.Context(), internalReq, nil)
	if err != nil {
		log.Printf("[Chat] Error: %v", err)
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		json.NewEncoder(w).Encode(map[string]any{
			"id":                 completionID,
			"object":             "chat.completion",
			"created":            createdTime,
			"model":              req.Model,
			"system_fingerprint": "fp_google_proxy",
			"choices": []any{
				map[string]any{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": fmt.Sprintf("[Error: %v]", err),
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     estimateTokens(internalMessages),
				"completion_tokens": 5,
				"total_tokens":      estimateTokens(internalMessages) + 5,
			},
		})
		return
	}

	if session != nil {
		session.Messages = append(session.Messages, providers.Message{
			Role:      "assistant",
			Content:   []providers.ContentPart{{Type: providers.ContentTypeText, Text: resp.Content}},
			ToolCalls: resp.ToolCalls,
		})
	}

	hasToolCalls := len(resp.ToolCalls) > 0
	finishReason := "stop"
	if hasToolCalls {
		finishReason = "tool_calls"
	}

	msgObj := map[string]any{"role": "assistant"}
	if hasToolCalls && resp.Content == "" {
		msgObj["content"] = nil
	} else {
		msgObj["content"] = resp.Content
	}

	if hasToolCalls {
		var openaiTCs []map[string]any
		for i, tc := range resp.ToolCalls {
			tcID := tc.ID
			if tcID == "" {
				tcID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), i)
			}
			tcArgs := tc.Function.Arguments
			if tcArgs == "" {
				tcArgs = "{}"
			}
			log.Printf("[DEBUG][PROXY -> CLIENT] Non-Stream Tool Call | Index: %d, ID: %s, Function: %s, Arguments: %s", i, tcID, tc.Function.Name, tcArgs)
			openaiTCs = append(openaiTCs, map[string]any{
				"id":   tcID,
				"type": "function",
				"function": map[string]any{
					"name":      tc.Function.Name,
					"arguments": tcArgs,
				},
			})
		}
		msgObj["tool_calls"] = openaiTCs
	}

	promptTokens := estimateTokens(internalMessages)
	completionTokens := estimateStringTokens(resp.Content)

	finalResp := map[string]any{
		"id":                 completionID,
		"object":             "chat.completion",
		"created":            createdTime,
		"model":              req.Model,
		"system_fingerprint": "fp_google_proxy",
		"choices": []any{
			map[string]any{
				"index":         0,
				"message":       msgObj,
				"finish_reason": finishReason,
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      promptTokens + completionTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(finalResp)
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	type openAIModel struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}

	path := strings.TrimPrefix(r.URL.Path, "/v1/models")
	path = strings.TrimPrefix(path, "/models")
	path = strings.TrimPrefix(path, "/")

	if path != "" {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		json.NewEncoder(w).Encode(openAIModel{
			ID:      path,
			Object:  "model",
			Created: 1700000000,
			OwnedBy: "google",
		})
		return
	}

	models, err := googleProvider.ListModels(r.Context())
	if err != nil {
		log.Printf("[API] Error listing models: %v", err)
	}

	var modelsList []openAIModel
	for _, m := range models {
		modelsList = append(modelsList, openAIModel{
			ID:      m.ID,
			Object:  "model",
			Created: 1700000000,
			OwnedBy: "google",
		})
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   modelsList,
	})
}

var (
	userRequestRegex = regexp.MustCompile(`(?s)<userRequest>\s*(.*?)\s*</userRequest>`)
	reminderRegex    = regexp.MustCompile(`(?s)<reminderInstructions>.*?</reminderInstructions>`)
)

func cleanUserPrompt(text string) string {
	matches := userRequestRegex.FindAllStringSubmatch(text, -1)
	if len(matches) > 0 {
		var extracted []string
		for _, m := range matches {
			if len(m) > 1 && strings.TrimSpace(m[1]) != "" {
				extracted = append(extracted, strings.TrimSpace(m[1]))
			}
		}
		if len(extracted) > 0 {
			return strings.Join(extracted, "\n\n")
		}
	}
	if reminderRegex.MatchString(text) {
		text = reminderRegex.ReplaceAllString(text, "")
		text = strings.TrimSpace(text)
	}
	return text
}

func parseOpenAIMessageParts(content any, images []string) []providers.ContentPart {
	var parts []providers.ContentPart

	if content != nil {
		switch c := content.(type) {
		case string:
			if c != "" {
				parts = append(parts, providers.ContentPart{
					Type: providers.ContentTypeText,
					Text: cleanUserPrompt(c),
				})
			}
		case []any:
			for _, item := range c {
				switch p := item.(type) {
				case string:
					if p != "" {
						parts = append(parts, providers.ContentPart{
							Type: providers.ContentTypeText,
							Text: cleanUserPrompt(p),
						})
					}
				case map[string]any:
					partType, _ := p["type"].(string)
					switch partType {
					case "text":
						if text, ok := p["text"].(string); ok && text != "" {
							parts = append(parts, providers.ContentPart{
								Type: providers.ContentTypeText,
								Text: cleanUserPrompt(text),
							})
						}
					case "image_url":
						var urlStr string
						if urlObj, ok := p["image_url"].(map[string]any); ok {
							if u, ok := urlObj["url"].(string); ok {
								urlStr = u
							}
						} else if u, ok := p["image_url"].(string); ok {
							urlStr = u
						}
						if urlStr != "" {
							if cp, err := parseImageURLToContentPart(urlStr); err == nil {
								parts = append(parts, cp)
							} else {
								log.Printf("[Vision] Error parsing image_url: %v", err)
							}
						}
					case "image":
						var imgStr string
						if s, ok := p["image"].(string); ok {
							imgStr = s
						} else if u, ok := p["url"].(string); ok {
							imgStr = u
						}
						if imgStr != "" {
							if cp, err := parseImageURLToContentPart(imgStr); err == nil {
								parts = append(parts, cp)
							}
						}
					default:
						if text, ok := p["text"].(string); ok && text != "" {
							parts = append(parts, providers.ContentPart{
								Type: providers.ContentTypeText,
								Text: cleanUserPrompt(text),
							})
						}
					}
				}
			}
		}
	}

	for _, img := range images {
		if cp, err := parseImageURLToContentPart(img); err == nil {
			parts = append(parts, cp)
		}
	}

	return parts
}

func parseImageURLToContentPart(imageRef string) (providers.ContentPart, error) {
	imageRef = strings.TrimSpace(imageRef)

	// 1. Data URI: data:image/png;base64,...
	if strings.HasPrefix(imageRef, "data:") {
		colonIdx := strings.Index(imageRef, ":")
		commaIdx := strings.Index(imageRef, ",")
		if commaIdx > colonIdx {
			meta := imageRef[colonIdx+1 : commaIdx]
			rawB64 := imageRef[commaIdx+1:]
			mimeType := "image/jpeg"
			if semiIdx := strings.Index(meta, ";"); semiIdx != -1 {
				mimeType = meta[:semiIdx]
			} else if meta != "" {
				mimeType = meta
			}

			decoded, err := base64.StdEncoding.DecodeString(rawB64)
			if err != nil {
				decoded, err = base64.URLEncoding.DecodeString(rawB64)
			}
			if err != nil {
				return providers.ContentPart{}, fmt.Errorf("failed to decode data uri base64: %w", err)
			}
			return providers.ContentPart{
				Type:     providers.ContentTypeImage,
				MimeType: mimeType,
				Data:     decoded,
			}, nil
		}
	}

	// 2. HTTP / HTTPS URL
	if strings.HasPrefix(imageRef, "http://") || strings.HasPrefix(imageRef, "https://") {
		client := &http.Client{Timeout: 10 * time.Second}
		resp, err := client.Get(imageRef)
		if err != nil {
			return providers.ContentPart{}, fmt.Errorf("failed to fetch image from URL %s: %w", imageRef, err)
		}
		defer resp.Body.Close()
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return providers.ContentPart{}, fmt.Errorf("failed to read image body from URL %s: %w", imageRef, err)
		}
		mimeType := resp.Header.Get("Content-Type")
		if mimeType == "" || !strings.HasPrefix(mimeType, "image/") {
			mimeType = http.DetectContentType(data)
		}
		if !strings.HasPrefix(mimeType, "image/") {
			mimeType = "image/jpeg"
		}
		return providers.ContentPart{
			Type:     providers.ContentTypeImage,
			MimeType: mimeType,
			Data:     data,
		}, nil
	}

	// 3. Raw Base64 string
	cleanedB64 := strings.ReplaceAll(strings.ReplaceAll(imageRef, "\n", ""), "\r", "")
	decoded, err := base64.StdEncoding.DecodeString(cleanedB64)
	if err == nil && len(decoded) > 0 {
		mimeType := http.DetectContentType(decoded)
		if !strings.HasPrefix(mimeType, "image/") {
			mimeType = "image/jpeg"
		}
		return providers.ContentPart{
			Type:     providers.ContentTypeImage,
			MimeType: mimeType,
			Data:     decoded,
		}, nil
	}

	return providers.ContentPart{}, fmt.Errorf("unknown image reference format")
}

func estimateStringTokens(s string) int {
	if s == "" {
		return 0
	}
	tokens := len(s) / 4
	if tokens == 0 {
		return 1
	}
	return tokens
}

func estimateTokens(messages []providers.Message) int {
	count := 0
	for _, m := range messages {
		count += 4
		for _, p := range m.Content {
			if p.Text != "" {
				count += estimateStringTokens(p.Text)
			}
			if len(p.Data) > 0 {
				count += 258
			}
		}
		for _, tc := range m.ToolCalls {
			count += estimateStringTokens(tc.Function.Name) + estimateStringTokens(tc.Function.Arguments)
		}
	}
	if count == 0 {
		return 1
	}
	return count
}
