package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Neuxbane/Ollama-One/providers"
)

type ConfigEntry struct {
	Type string `json:"provider"`
	Key  string `json:"key"`
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

type OpenAIToolCall struct {
	ID       string                 `json:"id"`
	Type     string                 `json:"type"`
	Function OllamaToolCallFunction `json:"function"`
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

type OpenAIChatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type OpenAIChatCompletionRequest struct {
	Model    string              `json:"model"`
	Messages []OpenAIChatMessage `json:"messages"`
	Stream   bool                `json:"stream"`
	Tools    []any               `json:"tools,omitempty"`
}

var (
	geminiProvider *providers.GeminiProvider
	openaiProvider *providers.OpenAIProvider
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
	var openaiKeys []string

	for _, entry := range config {
		switch entry.Type {
		case "gemini":
			geminiKeys = append(geminiKeys, entry.Key)
		case "openai":
			openaiKeys = append(openaiKeys, entry.Key)
		}
	}

	geminiProvider = &providers.GeminiProvider{
		BaseProvider: providers.BaseProvider{APIKeys: geminiKeys},
	}
	openaiProvider = &providers.OpenAIProvider{
		BaseProvider: providers.BaseProvider{APIKeys: openaiKeys},
	}

	if _, err := os.Stat("log"); os.IsNotExist(err) {
		os.Mkdir("log", 0755)
	}

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
		var allModels []OllamaModel
		collect := func(p providers.Provider, family string) {
			if p == nil {
				return
			}
			models, err := p.ListModels(r.Context())
			if err != nil {
				log.Printf("Error listing models for %s: %v", family, err)
				return
			}
			for _, m := range models {
				if len(allModels) >= 100 {
					break
				}
				// Create a real SHA256 digest based on the ID (not used anymore, using spoofed)

				allModels = append(allModels, OllamaModel{
					Name:       m.ID + ":latest",
					Model:      m.ID + ":latest",
					ModifiedAt: "2026-04-08T00:06:52.567291895+07:00",
					Size:       4683087332,
					Digest:     "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e",
					Details: ModelDetail{
						ParentModel:       "",
						Format:            "gguf",
						Family:            "qwen2",
						Families:          []string{"qwen2"},
						ParameterSize:     "7.6B",
						QuantizationLevel: "Q4_K_M",
					},
				})
			}
		}
		if len(geminiProvider.APIKeys) > 0 {
			collect(geminiProvider, "gemini")
		}
		if len(openaiProvider.APIKeys) > 0 {
			collect(openaiProvider, "openai")
		}
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		data, _ := json.Marshal(OllamaTagsResponse{Models: allModels})
		w.Write(data)
	})

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

	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req OpenAIChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		for _, m := range req.Messages {
			content := extractOpenAITextContent(m.Content)
			log.Printf("Client -> Proxy: [%s] %s", m.Role, content)
		}
		req.Model = normalizeModelName(req.Model)

		provider := getProvider(req.Model)
		internalMessages := make([]providers.Message, len(req.Messages))
		for i, m := range req.Messages {
			internalMessages[i] = providers.Message{
				Role: m.Role,
				Content: []providers.ContentPart{
					{Type: providers.ContentTypeText, Text: extractOpenAITextContent(m.Content)},
				},
			}
		}

		internalReq := &providers.CompletionRequest{
			Model:    req.Model,
			Messages: internalMessages,
			Stream:   req.Stream,
		}

		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")

			flusher, ok := w.(http.Flusher)
			_, err := provider.Chat(r.Context(), internalReq, func(chunk *providers.CompletionResponse) {
				chunkResp := map[string]any{
					"id":      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
					"object":  "chat.completion.chunk",
					"created": time.Now().Unix(),
					"model":   req.Model,
					"choices": []any{
						map[string]any{
							"index": 0,
							"delta": map[string]any{"content": chunk.Content},
							"finish_reason": nil,
						},
					},
				}
				
				if len(chunk.ToolCalls) > 0 {
					var openaiTCs []map[string]any
					for i, tc := range chunk.ToolCalls {
						openaiTCs = append(openaiTCs, map[string]any{
							"index": i,
							"id": tc.ID,
							"type": "function",
							"function": map[string]any{
								"name": tc.Function.Name,
								"arguments": tc.Function.Arguments,
							},
						})
					}
					// Update delta to include tool_calls
					choices := chunkResp["choices"].([]any)
					choice := choices[0].(map[string]any)
					delta := choice["delta"].(map[string]any)
					delta["tool_calls"] = openaiTCs
				}

				data, _ := json.Marshal(chunkResp)
				fmt.Fprintf(w, "data: %s\n\n", data)
				log.Printf("Provider -> Proxy: %s", chunk.Content)
				log.Printf("Proxy -> Client: %s", chunk.Content)
				if ok {
					flusher.Flush()
				}
			})
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}

			endResp := map[string]any{
				"id":      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
				"object":  "chat.completion.chunk",
				"created": time.Now().Unix(),
				"model":   req.Model,
				"choices": []any{
					map[string]any{
						"index": 0,
						"delta": map[string]any{},
						"finish_reason": "stop",
					},
				},
			}

			endData, _ := json.Marshal(endResp)
			fmt.Fprintf(w, "data: %s\n\n", endData)
			fmt.Fprint(w, "data: [DONE]\n\n")
			log.Printf("Provider -> Proxy: [DONE]")
			log.Printf("Proxy -> Client: [DONE]")
			if ok {
				flusher.Flush()
			}
			return
		}

		resp, err := provider.Chat(r.Context(), internalReq, nil)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		finalResp := map[string]any{
			"id":      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   req.Model,
			"choices": []any{
				map[string]any{
					"index": 0,
					"message": map[string]any{
						"role":    "assistant",
						"content": resp.Content,
					},
					"finish_reason": "stop",
				},
			},
		}

		if len(resp.ToolCalls) > 0 {
			var openaiTCs []map[string]any
			for _, tc := range resp.ToolCalls {
				openaiTCs = append(openaiTCs, map[string]any{
					"id": tc.ID,
					"type": "function",
					"function": map[string]any{
						"name": tc.Function.Name,
						"arguments": tc.Function.Arguments,
					},
				})
			}
			choices := finalResp["choices"].([]any)
			choice := choices[0].(map[string]any)
			msg := choice["message"].(map[string]any)
			msg["tool_calls"] = openaiTCs
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(finalResp)
		log.Printf("Provider -> Proxy: %s", resp.Content)
		log.Printf("Proxy -> Client: [FULL RESPONSE]")
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "Ollama is running")
	})

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, X-Requested-With, Ollama-Version, X-Ollama-Version")
		w.Header().Set("Access-Control-Expose-Headers", "Ollama-Version, X-Ollama-Version")
		w.Header().Set("Ollama-Version", "0.11.8")
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if r.URL.Path == "/favicon.ico" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		mux.ServeHTTP(w, r)
	})

	fmt.Println("Ollama-One proxy starting on 127.0.0.1:11434...")
	if err := http.ListenAndServe("127.0.0.1:11434", handler); err != nil {
		fmt.Printf("Error starting server: %v\n", err)
	}
}

func getProvider(model string) providers.Provider {
	model = normalizeModelName(model)
	switch {
	case model == "gpt-4" || model == "gpt-4o" || model == "gpt-3.5-turbo":
		return openaiProvider
	case strings.HasPrefix(model, "gemini-") || strings.HasPrefix(model, "gemma-") || model == "gemma-":
		return geminiProvider
	default:
		log.Printf("Warning: Model %q is not explicitly mapped to a provider, defaulting to OpenAI", model)
		return openaiProvider
	}
}

func normalizeModelName(model string) string {
	model = strings.TrimSpace(model)
	return strings.TrimSuffix(model, ":latest")
}

func handleGenerate(w http.ResponseWriter, r *http.Request, req *OllamaGenerateRequest) {
	provider := getProvider(req.Model)
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
	provider := getProvider(req.Model)

	internalMessages := make([]providers.Message, len(req.Messages))
	for i, m := range req.Messages {
		parts := []providers.ContentPart{
			{
				Type: providers.ContentTypeText,
				Text: m.Content,
			},
		}
		for _, img := range m.Images {
			parts = append(parts, providers.ContentPart{
				Type:     providers.ContentTypeImage,
				MimeType: "image/jpeg",
				Data:     []byte(img),
			})
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
}

func extractOpenAITextContent(content any) string {
	switch v := content.(type) {
	case string:
		return v
	case []any:
		var b strings.Builder
		for _, item := range v {
			part, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if part["type"] == "text" {
				if text, ok := part["text"].(string); ok {
					b.WriteString(text)
				}
			}
		}
		return b.String()
	default:
		return ""
	}
}
