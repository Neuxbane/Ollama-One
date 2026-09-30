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

type OllamaToolCallFunction struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type OllamaToolCall struct {
	ID       string                 `json:"id,omitempty"`
	Type     string                 `json:"type,omitempty"`
	Function OllamaToolCallFunction `json:"function"`
}

type loggingResponseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (lrw *loggingResponseWriter) WriteHeader(code int) {
	lrw.statusCode = code
	lrw.ResponseWriter.WriteHeader(code)
}

type OllamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Images    []string         `json:"images,omitempty"`
	ToolCalls []OllamaToolCall `json:"tool_calls,omitempty"`
}

type OllamaTool struct {
	Type     string         `json:"type"`
	Function map[string]any `json:"function"`
}

type OllamaChatRequest struct {
	Model     string          `json:"model"`
	Messages  []OllamaMessage `json:"messages"`
	Stream    bool            `json:"stream"`
	Tools     []OllamaTool    `json:"tools,omitempty"`
	SessionID string          `json:"session_id,omitempty"`
	Thinking  *providers.ThinkingConfig `json:"thinking_config,omitempty"`
}

var sessionManager = NewSessionManager()

type OllamaChatResponse struct {
	Model     string        `json:"model"`
	CreatedAt time.Time     `json:"created_at"`
	Message   OllamaMessage `json:"message"`
	Done      bool          `json:"done"`
}

type OllamaVersionResponse struct {
	Version string `json:"version"`
}

type OllamaModel struct {
	Name       string      `json:"name"`
	Model      string      `json:"model"`
	ModifiedAt string      `json:"modified_at"`
	Size       int64       `json:"size"`
	Digest     string      `json:"digest"`
	Details    ModelDetail `json:"details"`
}

type ModelDetail struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

type OllamaTagsResponse struct {
	Models []OllamaModel `json:"models"`
}

type OllamaGenerateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type OllamaGenerateResponse struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Response  string `json:"response"`
	Done      bool   `json:"done"`
}

type OllamaShowRequest struct {
	Name  string `json:"name"`
	Model string `json:"model"` // Some clients might use 'model' instead of 'name'
}

type OllamaShowResponse struct {
	License      string         `json:"license"`
	Modelfile    string         `json:"modelfile"`
	Template     string         `json:"template"`
	System       string         `json:"system"`
	Details      ModelDetail    `json:"details"`
	Capabilities []string       `json:"capabilities"`
	ModifiedAt   string         `json:"modified_at"`
	ModelInfo    map[string]any `json:"model_info"`
	Tensors      []any          `json:"tensors"`
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
	geminiProvider     *providers.GeminiProvider
	geminiLiveProvider *providers.GeminiLiveProvider
	openaiProvider     *providers.OpenAIProvider
	providersMap       = make(map[string]providers.Provider)
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

	var geminiKeys []string
	var geminiLiveKeys []string
	var openaiKeys []string

	for _, entry := range config {
		t := strings.ToLower(entry.GetType())
		switch t {
		case "gemini":
			geminiKeys = append(geminiKeys, entry.Key)
			geminiLiveKeys = append(geminiLiveKeys, entry.Key)
		case "gemini-live", "live", "geminilive":
			geminiLiveKeys = append(geminiLiveKeys, entry.Key)
			geminiKeys = append(geminiKeys, entry.Key)
		case "openai":
			openaiKeys = append(openaiKeys, entry.Key)
		default:
			log.Printf("[CONFIG] Warning: unknown provider type %q in config", entry.GetType())
		}
	}

	// Share keys between Gemini and Gemini Live if either is missing
	if len(geminiLiveKeys) == 0 && len(geminiKeys) > 0 {
		geminiLiveKeys = append(geminiLiveKeys, geminiKeys...)
	}
	if len(geminiKeys) == 0 && len(geminiLiveKeys) > 0 {
		geminiKeys = append(geminiKeys, geminiLiveKeys...)
	}
	// Fallback to GEMINI_API_KEY environment variable if keys are empty
	if envKey := os.Getenv("GEMINI_API_KEY"); envKey != "" {
		if len(geminiKeys) == 0 {
			geminiKeys = append(geminiKeys, envKey)
		}
		if len(geminiLiveKeys) == 0 {
			geminiLiveKeys = append(geminiLiveKeys, envKey)
		}
	}
	// Fallback to OPENAI_API_KEY environment variable if keys are empty
	if envKey := os.Getenv("OPENAI_API_KEY"); envKey != "" {
		if len(openaiKeys) == 0 {
			openaiKeys = append(openaiKeys, envKey)
		}
	}

	log.Printf("[CONFIG] Loaded %d Gemini key(s), %d Gemini Live key(s), %d OpenAI key(s)", len(geminiKeys), len(geminiLiveKeys), len(openaiKeys))

	if len(geminiKeys) > 0 {
		geminiProvider = &providers.GeminiProvider{
			BaseProvider: providers.BaseProvider{APIKeys: geminiKeys},
		}
		providersMap["gemini"] = geminiProvider
	}
	if len(geminiLiveKeys) > 0 {
		geminiLiveProvider = providers.NewGeminiLiveProvider(geminiLiveKeys)
		providersMap["gemini-live"] = geminiLiveProvider
	}
	if len(openaiKeys) > 0 {
		openaiProvider = &providers.OpenAIProvider{
			BaseProvider: providers.BaseProvider{APIKeys: openaiKeys},
		}
		providersMap["openai"] = openaiProvider
	}


	// Log directory creation commented out as requested
	/*
	if _, err := os.Stat("log"); os.IsNotExist(err) {
		os.Mkdir("log", 0755)
	}
	*/

	mux := http.NewServeMux()

	mux.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		var req OllamaChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		for _, m := range req.Messages {
			log.Printf("Client -> Proxy: [%s] %s", m.Role, m.Content)
			if len(m.ToolCalls) > 0 {
				log.Printf("Client -> Proxy: [tool_calls] %d calls", len(m.ToolCalls))
			}
		}
		req.Model = normalizeModelName(req.Model)
		handleChat(w, r, &req)
	})

	mux.HandleFunc("/api/generate", func(w http.ResponseWriter, r *http.Request) {
		var req OllamaGenerateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		log.Printf("Client -> Proxy: [prompt] %s", req.Prompt)
		req.Model = normalizeModelName(req.Model)
		handleGenerate(w, r, &req)
	})

	mux.HandleFunc("/api/version", func(w http.ResponseWriter, r *http.Request) {
		resp := OllamaVersionResponse{Version: "0.11.8"}
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.Header().Set("Ollama-Version", "0.11.8")
		data, _ := json.Marshal(resp)
		w.Write(data)
	})

	mux.HandleFunc("/api/tags", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("[API] Listing models (/api/tags)...")
		var allModels []OllamaModel
		for pName, p := range providersMap {
			models, err := p.ListModels(r.Context())
			if err != nil {
				log.Printf("[API] Error listing models for provider %s: %v", pName, err)
				continue
			}
			log.Printf("[API] Provider %s returned %d models", pName, len(models))
			for _, m := range models {
				if len(allModels) >= 200 {
					break
				}
				namespacedName := fmt.Sprintf("%s/%s:latest", pName, m.ID)
				allModels = append(allModels, OllamaModel{
					Name:       namespacedName,
					Model:      namespacedName,
					ModifiedAt: "2026-04-08T00:06:52.567291895+07:00",
					Size:       4683087332,
					Digest:     "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e",
					Details: ModelDetail{
						ParentModel:       "",
						Format:            "gguf",
						Family:            pName,
						Families:          []string{pName},
						ParameterSize:     "7.6B",
						QuantizationLevel: "Q4_K_M",
					},
				})
				bareName := m.ID + ":latest"
				allModels = append(allModels, OllamaModel{
					Name:       bareName,
					Model:      bareName,
					ModifiedAt: "2026-04-08T00:06:52.567291895+07:00",
					Size:       4683087332,
					Digest:     "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e",
					Details: ModelDetail{
						ParentModel:       "",
						Format:            "gguf",
						Family:            pName,
						Families:          []string{pName},
						ParameterSize:     "7.6B",
						QuantizationLevel: "Q4_K_M",
					},
				})
			}
		}
		log.Printf("[API] /api/tags returning %d model(s)", len(allModels))
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		data, _ := json.Marshal(OllamaTagsResponse{Models: allModels})
		w.Write(data)
	})

	mux.HandleFunc("/api/ps", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.Write([]byte(`{"models":[]}`))
	})

	mux.HandleFunc("/v1/models", handleOpenAIModels)
	mux.HandleFunc("/models", handleOpenAIModels)
	mux.HandleFunc("/v1/models/", handleOpenAIModels)
	mux.HandleFunc("/models/", handleOpenAIModels)

	mux.HandleFunc("/api/show", func(w http.ResponseWriter, r *http.Request) {
		var req OllamaShowRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		// Fallback to Model if Name is empty
		if req.Name == "" && req.Model != "" {
			req.Name = req.Model
		}

		id := strings.TrimSuffix(req.Name, ":latest")
		family := "gemini"
		if strings.HasPrefix(id, "gpt-") {
			family = "openai"
		} else if strings.Contains(id, "live") || strings.Contains(id, "native-audio") {
			family = "gemini-live"
		}

		resp := OllamaShowResponse{
			License:   "",
			Modelfile: "FROM " + req.Name,
			Template:  "{{ .System }}\n{{ .Prompt }}",
			Details: ModelDetail{
				ParentModel:       "",
				Format:            "gguf",
				Family:            family,
				Families:          []string{family},
				ParameterSize:     "unknown",
				QuantizationLevel: "Q4_0",
			},
			Capabilities: []string{"chat", "completion", "vision", "tools"},
			ModifiedAt:   time.Now().Format(time.RFC3339Nano),
			ModelInfo: map[string]any{
				"general.architecture": family,
			},
			Tensors: []any{},
		}
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		data, _ := json.Marshal(resp)
		w.Write(data)
	})

	mux.HandleFunc("/v1/chat/completions", handleOpenAIChatCompletions)
	mux.HandleFunc("/chat/completions", handleOpenAIChatCompletions)

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "Ollama is running")
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
		w.Header().Set("Access-Control-Expose-Headers", "Ollama-Version, X-Ollama-Version")
		w.Header().Set("Ollama-Version", "0.11.8")

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
			fmt.Printf("Ollama-One proxy starting on http://%s...\n", addr)
			if err := http.Serve(listener, handler); err != nil {
				fmt.Printf("Error running server: %v\n", err)
			}
			return
		}
		fmt.Printf("Port %s unavailable (%v), trying next port...\n", p, err)
	}
}

func getProvider(model string) (providers.Provider, string) {
	fullModel := strings.TrimSpace(model)
	fullModel = strings.TrimSuffix(fullModel, ":latest")

	// 1. Explicit Provider Namespace: {provider}/{modelName} or {provider}:{modelName}
	// e.g. "gemini/gemini-2.0-flash", "openai/gpt-4o", "gemini-live/gemini-3.1-flash-live-preview", "live/gemini-3.1-flash-live-preview"
	if idx := strings.IndexAny(fullModel, "/:"); idx != -1 {
		providerPrefix := strings.ToLower(fullModel[:idx])
		actualModelName := fullModel[idx+1:]
		if (providerPrefix == "live" || providerPrefix == "gemini-live") && geminiLiveProvider != nil {
			log.Printf("[ROUTE] Explicit namespace %q -> Provider: gemini-live, Target Model: %s", model, actualModelName)
			return geminiLiveProvider, actualModelName
		}
		if p, ok := providersMap[providerPrefix]; ok {
			log.Printf("[ROUTE] Explicit namespace %q -> Provider: %s, Target Model: %s", model, providerPrefix, actualModelName)
			return p, actualModelName
		}
	}

	// 2. Gemini Live model detection by model name pattern
	modelLower := strings.ToLower(fullModel)
	if geminiLiveProvider != nil && (strings.Contains(modelLower, "live") || strings.Contains(modelLower, "native-audio")) {
		log.Printf("[ROUTE] Live model pattern match -> Provider: gemini-live, Model: %s", fullModel)
		return geminiLiveProvider, fullModel
	}

	// 3. OpenAI model pattern match
	if strings.HasPrefix(modelLower, "gpt-") || strings.HasPrefix(modelLower, "o1") || strings.HasPrefix(modelLower, "o3") || strings.HasPrefix(modelLower, "chatgpt") {
		if openaiProvider != nil {
			log.Printf("[ROUTE] OpenAI model pattern match -> Provider: openai, Model: %s", fullModel)
			return openaiProvider, fullModel
		}
		// If openaiProvider not configured, alias to geminiProvider so it doesn't fail
		if geminiProvider != nil {
			log.Printf("[ROUTE] OpenAI model %q requested but no OpenAI key; routing to Gemini -> Model: gemini-2.0-flash", fullModel)
			return geminiProvider, "gemini-2.0-flash"
		}
	}

	// 4. Gemini model pattern match
	if strings.HasPrefix(modelLower, "gemini") || strings.HasPrefix(modelLower, "learnlm") {
		if geminiProvider != nil {
			log.Printf("[ROUTE] Gemini model pattern match -> Provider: gemini, Model: %s", fullModel)
			return geminiProvider, fullModel
		}
	}

	// 5. Single active provider: route everything to it
	if len(providersMap) == 1 {
		for pName, p := range providersMap {
			log.Printf("[ROUTE] Single active provider %q -> Model: %s", pName, fullModel)
			return p, fullModel
		}
	}

	// 6. Provider name prefix match
	for pName, p := range providersMap {
		if strings.HasPrefix(modelLower, pName) {
			log.Printf("[ROUTE] Provider prefix match -> Provider: %s, Model: %s", pName, fullModel)
			return p, fullModel
		}
	}

	// 7. Default fallback to registered provider
	if geminiProvider != nil {
		log.Printf("[ROUTE] Fallback -> Provider: gemini, Model: %s", fullModel)
		return geminiProvider, fullModel
	}
	for pName, p := range providersMap {
		log.Printf("[ROUTE] Fallback -> Provider: %s, Model: %s", pName, fullModel)
		return p, fullModel
	}

	log.Printf("[ROUTE] Warning: No active provider found for model %q, defaulting to geminiProvider", fullModel)
	return geminiProvider, fullModel
}

func normalizeModelName(model string) string {
	model = strings.TrimSpace(model)
	return strings.TrimSuffix(model, ":latest")
}

func handleGenerate(w http.ResponseWriter, r *http.Request, req *OllamaGenerateRequest) {
	provider, targetModel := getProvider(req.Model)
	req.Model = targetModel
	internalReq := &providers.CompletionRequest{
		Model: req.Model,
		Messages: []providers.Message{
			{
				Role: "user",
				Content: []providers.ContentPart{
					{Type: providers.ContentTypeText, Text: req.Prompt},
				},
			},
		},
		Stream: req.Stream,
	}

	if req.Stream {
		w.Header().Set("Content-Type", "application/json")
		flusher, ok := w.(http.Flusher)
		var fullResponse providers.CompletionResponse
		_, err := provider.Chat(r.Context(), internalReq, func(chunk *providers.CompletionResponse) {
			fullResponse.Content += chunk.Content
			resp := OllamaGenerateResponse{
				Model:    req.Model,
				Response: chunk.Content,
				Done:     false,
			}
			json.NewEncoder(w).Encode(resp)
			log.Printf("Provider -> Proxy: %s", chunk.Content)
			if ok {
				flusher.Flush()
			}
		})
		if err == nil {
			finalResp := OllamaGenerateResponse{Model: req.Model, Done: true}
			json.NewEncoder(w).Encode(finalResp)
			log.Printf("Proxy -> Client: [DONE]")
			logInteraction(req.Model, []OllamaMessage{{Role: "user", Content: req.Prompt}}, nil, &fullResponse, finalResp)
		}
	} else {
		resp, err := provider.Chat(r.Context(), internalReq, nil)
		if err == nil {
			w.Header().Set("Content-Type", "application/json")
			finalResp := OllamaGenerateResponse{
				Model:    req.Model,
				Response: resp.Content,
				Done:     true,
			}
			json.NewEncoder(w).Encode(finalResp)
			log.Printf("Provider -> Proxy: %s", resp.Content)
			log.Printf("Proxy -> Client: [FULL RESPONSE]")
			logInteraction(req.Model, []OllamaMessage{{Role: "user", Content: req.Prompt}}, nil, resp, finalResp)
		} else {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

func handleChat(w http.ResponseWriter, r *http.Request, req *OllamaChatRequest) {
	provider, targetModel := getProvider(req.Model)
	req.Model = targetModel

	internalMessages := make([]providers.Message, len(req.Messages))
	for i, m := range req.Messages {
		parts := []providers.ContentPart{
			{
				Type: providers.ContentTypeText,
				Text: m.Content,
			},
		}
		for _, img := range m.Images {
			if cp, err := parseImageURLToContentPart(img); err == nil {
				parts = append(parts, cp)
			} else {
				parts = append(parts, providers.ContentPart{
					Type:     providers.ContentTypeImage,
					MimeType: "image/jpeg",
					Data:     []byte(img),
				})
			}
		}
		internalMessages[i] = providers.Message{
			Role:    m.Role,
			Content: parts,
		}
	}

	// Convert OllamaTool objects to providers.Tool objects
	var internalTools []providers.Tool
	for _, tool := range req.Tools {
		t := providers.Tool{
			Type: tool.Type,
		}
		if tool.Function != nil {
			// Extract function details from the map
			funcTool := providers.Tool{
				Type: "function",
			}
			if name, ok := tool.Function["name"].(string); ok {
				funcTool.Name = name
			}
			if desc, ok := tool.Function["description"].(string); ok {
				funcTool.Description = desc
			}
			if params, ok := tool.Function["parameters"].(map[string]any); ok {
				funcTool.Parameters = params
			}
			t.Functions = append(t.Functions, funcTool)
		}
		internalTools = append(internalTools, t)
	}

	internalReq := &providers.CompletionRequest{
		Model:    req.Model,
		Messages: internalMessages,
		Stream:   req.Stream,
		Tools:    internalTools,
		Thinking: req.Thinking,
	}

	// Handle Sessions
	var session *Session
	if req.SessionID != "" {
		session = sessionManager.GetSession(req.SessionID)
		
		// If client sends messages, we try to detect if it's a new turn
		if len(internalReq.Messages) > 0 {
			lastClientMsg := internalReq.Messages[len(internalReq.Messages)-1]
			
			// If session is empty, just use client's messages
			if len(session.Messages) == 0 {
				session.Messages = internalReq.Messages
			} else {
				// Check if the last client message is already in session
				// If not, it's a new message from the user
				found := false
				for _, m := range session.Messages {
					if m.Role == lastClientMsg.Role && len(m.Content) > 0 && len(lastClientMsg.Content) > 0 && m.Content[0].Text == lastClientMsg.Content[0].Text {
						found = true
						break
					}
				}
				
				if !found {
					session.Messages = append(session.Messages, lastClientMsg)
				}
			}
			// Use session messages for the actual request
			internalReq.Messages = session.Messages
			log.Printf("Using session %s history (length: %d)", req.SessionID, len(internalReq.Messages))
		}
	}

	if req.Stream {
		var fullResponse providers.CompletionResponse
		flusher, ok := w.(http.Flusher)
		_, err := provider.Chat(r.Context(), internalReq, func(chunk *providers.CompletionResponse) {
			fullResponse.Content += chunk.Content
			if len(chunk.ToolCalls) > 0 {
				fullResponse.ToolCalls = append(fullResponse.ToolCalls, chunk.ToolCalls...)
			}

			var ollamaTCs []OllamaToolCall
			if len(chunk.ToolCalls) > 0 {
				for _, tc := range chunk.ToolCalls {
					var args map[string]any
					if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
						args = make(map[string]any)
					}
					log.Printf("Provider -> Proxy (Tool Call): %s(%v)", tc.Function.Name, args)
					ollamaTCs = append(ollamaTCs, OllamaToolCall{
						ID:   tc.ID,
						Type: tc.Type,
						Function: OllamaToolCallFunction{
							Name:      tc.Function.Name,
							Arguments: args,
						},
					})
				}
			}

			respChunk := OllamaChatResponse{
				Model:     req.Model,
				CreatedAt: time.Now(),
				Message: OllamaMessage{
					Role:      "assistant",
					Content:   chunk.Content,
					ToolCalls: ollamaTCs,
				},
				Done: false,
			}
			json.NewEncoder(w).Encode(respChunk)
			log.Printf("Proxy -> Client: %s", chunk.Content)
			if ok {
				flusher.Flush()
			}
		})

		if err == nil {
			// Update Session with full assistant response
			if session != nil {
				assistantMsg := providers.Message{
					Role:      "assistant",
					Content:   []providers.ContentPart{{Type: providers.ContentTypeText, Text: fullResponse.Content}},
					ToolCalls: fullResponse.ToolCalls,
				}
				session.Messages = append(session.Messages, assistantMsg)
				log.Printf("Updated session %s with assistant message (length: %d)", session.ID, len(session.Messages))
			}

			var finalOllamaTCs []OllamaToolCall
			for _, tc := range fullResponse.ToolCalls {
				var args map[string]any
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
					args = make(map[string]any)
				}
				finalOllamaTCs = append(finalOllamaTCs, OllamaToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: OllamaToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: args,
					},
				})
			}

			finalResp := OllamaChatResponse{
				Model:     req.Model,
				CreatedAt: time.Now(),
				Message: OllamaMessage{
					Role:      "assistant",
					Content:   "",
					ToolCalls: finalOllamaTCs,
				},
				Done: true,
			}
			json.NewEncoder(w).Encode(finalResp)
			
			log.Printf("Proxy -> Client: [DONE]")
			logInteraction(req.Model, req.Messages, req.Tools, &fullResponse, finalResp)
		} else {
			log.Printf("Provider error: %v", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	} else {
		resp, err := provider.Chat(r.Context(), internalReq, nil)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		
		// Update Session
		if session != nil {
			assistantMsg := providers.Message{
				Role:      "assistant",
				Content:   []providers.ContentPart{{Type: providers.ContentTypeText, Text: resp.Content}},
				ToolCalls: resp.ToolCalls,
			}
			session.Messages = append(session.Messages, assistantMsg)
		}

		var ollamaTCs []OllamaToolCall
		for _, tc := range resp.ToolCalls {
			var args map[string]any
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				args = make(map[string]any)
			}
			ollamaTCs = append(ollamaTCs, OllamaToolCall{
				ID:   tc.ID,
				Type: tc.Type,
				Function: OllamaToolCallFunction{
					Name:      tc.Function.Name,
					Arguments: args,
				},
			})
		}

		w.Header().Set("Content-Type", "application/json")
		finalResp := OllamaChatResponse{
			Model:     req.Model,
			CreatedAt: time.Now(),
			Message: OllamaMessage{
				Role:      "assistant",
				Content:   resp.Content,
				ToolCalls: ollamaTCs,
			},
			Done: true,
		}
		json.NewEncoder(w).Encode(finalResp)
		log.Printf("Provider -> Proxy: %s", resp.Content)
		log.Printf("Proxy -> Client: [FULL RESPONSE]")
		logInteraction(req.Model, req.Messages, req.Tools, resp, finalResp)
	}
}

func logInteraction(model string, messages []OllamaMessage, tools []OllamaTool, providerResponse *providers.CompletionResponse, clientResponse any) {
	// Interaction logging to file commented out as requested
	/*
	timestamp := time.Now().Format("20060102_150405")
	filename := filepath.Join("log", fmt.Sprintf("chat_%s_%s.log", timestamp, strings.ReplaceAll(model, ":", "_")))

	var logData struct {
		Timestamp time.Time `json:"timestamp"`
		Model     string    `json:"model"`
		ClientRequest struct {
			Messages []OllamaMessage `json:"messages"`
			Tools    []OllamaTool    `json:"tools,omitempty"`
		} `json:"client_request"`
		ProviderResponse *providers.CompletionResponse `json:"provider_response"`
		ClientResponse   any                          `json:"client_response"`
	}
	logData.Timestamp = time.Now()
	logData.Model = model
	logData.ClientRequest.Messages = messages
	logData.ClientRequest.Tools = tools
	logData.ProviderResponse = providerResponse
	logData.ClientResponse = clientResponse

	data, err := json.MarshalIndent(logData, "", "  ")
	if err != nil {
		log.Printf("Error marshaling log data: %v", err)
		return
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		log.Printf("Error writing log file: %v", err)
	}
	*/
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

func handleOpenAIChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req OpenAIChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	if req.Model == "" {
		req.Model = "gemini-2.0-flash"
	}

	req.Model = normalizeModelName(req.Model)
	provider, targetModel := getProvider(req.Model)
	req.Model = targetModel

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

		textContent := extractOpenAITextContent(m.Content)
		log.Printf("Client -> Proxy (OpenAI): [%s] %s (parts: %d, tool_calls: %d)", m.Role, textContent, len(parts), len(tcs))
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

		// Emit initial role chunk immediately so clients like VS Code Copilot receive choices right away
		initChunk := map[string]any{
			"id":                 completionID,
			"object":             "chat.completion.chunk",
			"created":            createdTime,
			"model":              req.Model,
			"system_fingerprint": "fp_ollama_one",
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

		_, err := provider.Chat(r.Context(), internalReq, func(chunk *providers.CompletionResponse) {
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
				"system_fingerprint": "fp_ollama_one",
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
			log.Printf("[OpenAI] Streaming chat error: %v", err)
			errChunk := map[string]any{
				"id":                 completionID,
				"object":             "chat.completion.chunk",
				"created":            createdTime,
				"model":              req.Model,
				"system_fingerprint": "fp_ollama_one",
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
			"system_fingerprint": "fp_ollama_one",
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
				"system_fingerprint": "fp_ollama_one",
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

	resp, err := provider.Chat(r.Context(), internalReq, nil)
	if err != nil {
		log.Printf("[OpenAI] Chat completion error: %v", err)
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		json.NewEncoder(w).Encode(map[string]any{
			"id":                 completionID,
			"object":             "chat.completion",
			"created":            createdTime,
			"model":              req.Model,
			"system_fingerprint": "fp_ollama_one",
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

	msgObj := map[string]any{
		"role": "assistant",
	}
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
		"system_fingerprint": "fp_ollama_one",
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

func handleOpenAIModels(w http.ResponseWriter, r *http.Request) {
	log.Printf("[API] Handling models request: %s %s", r.Method, r.URL.Path)
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
		modelID := path
		providerName := "ollama-one"
		if idx := strings.Index(modelID, "/"); idx != -1 {
			providerName = modelID[:idx]
		}
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		json.NewEncoder(w).Encode(openAIModel{
			ID:      modelID,
			Object:  "model",
			Created: 1700000000,
			OwnedBy: providerName,
		})
		return
	}

	var modelsList []openAIModel
	for pName, p := range providersMap {
		models, err := p.ListModels(r.Context())
		if err != nil {
			log.Printf("[API] Error listing models for provider %s: %v", pName, err)
			continue
		}
		for _, m := range models {
			modelsList = append(modelsList, openAIModel{
				ID:      fmt.Sprintf("%s/%s", pName, m.ID),
				Object:  "model",
				Created: 1700000000,
				OwnedBy: pName,
			})
			modelsList = append(modelsList, openAIModel{
				ID:      m.ID,
				Object:  "model",
				Created: 1700000000,
				OwnedBy: pName,
			})
		}
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   modelsList,
	})
}

func parseOpenAIMessageParts(content any, images []string) []providers.ContentPart {
	var parts []providers.ContentPart

	if content != nil {
		switch c := content.(type) {
		case string:
			if c != "" {
				parts = append(parts, providers.ContentPart{
					Type: providers.ContentTypeText,
					Text: c,
				})
			}
		case []any:
			for _, item := range c {
				switch p := item.(type) {
				case string:
					if p != "" {
						parts = append(parts, providers.ContentPart{
							Type: providers.ContentTypeText,
							Text: p,
						})
					}
				case map[string]any:
					partType, _ := p["type"].(string)
					switch partType {
					case "text":
						if text, ok := p["text"].(string); ok && text != "" {
							parts = append(parts, providers.ContentPart{
								Type: providers.ContentTypeText,
								Text: text,
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
								Text: text,
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
