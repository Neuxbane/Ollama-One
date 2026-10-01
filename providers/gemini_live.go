package providers

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/gorilla/websocket"
)

// SupportedLiveModels lists the preconfigured Gemini Live models
var SupportedLiveModels = []ModelInfo{
	{
		ID:           "gemini-3.8-live-extended-thinking",
		Name:         "Gemini 3.8 Flash Live Extended Thinking",
		ContextSize:  128000,
		Capabilities: []string{"audio", "vision", "tools", "chat", "completion"},
	},
	{
		ID:           "gemini-3.8-live",
		Name:         "Gemini 3.8 Flash Live",
		ContextSize:  128000,
		Capabilities: []string{"audio", "vision", "tools", "chat", "completion"},
	},
	{
		ID:           "gemini-3.1-flash-live-preview",
		Name:         "Gemini 3.1 Flash Live Preview",
		ContextSize:  128000,
		Capabilities: []string{"audio", "vision", "tools", "chat", "completion"},
	},
	{
		ID:           "gemini-2.5-flash-native-audio-preview-09-2025",
		Name:         "Gemini 2.5 Flash Native Audio Preview (09-2025)",
		ContextSize:  128000,
		Capabilities: []string{"audio", "vision", "tools", "chat", "completion"},
	},
	{
		ID:           "gemini-2.5-flash-native-audio-preview-12-2025",
		Name:         "Gemini 2.5 Flash Native Audio Preview (12-2025)",
		ContextSize:  128000,
		Capabilities: []string{"audio", "vision", "tools", "chat", "completion"},
	},
}

// WriteWavHeader writes a standard 44-byte RIFF/WAVE header for linear PCM data
func WriteWavHeader(buf []byte, sampleRate int, numChannels int, bitsPerSample int) []byte {
	header := make([]byte, 44)
	// RIFF chunk descriptor
	copy(header[0:4], "RIFF")
	binary.LittleEndian.PutUint32(header[4:8], uint32(len(buf)+36))
	copy(header[8:12], "WAVE")

	// "fmt " sub-chunk
	copy(header[12:16], "fmt ")
	binary.LittleEndian.PutUint32(header[16:20], 16) // Subchunk1Size for PCM
	binary.LittleEndian.PutUint16(header[20:22], 1)  // AudioFormat 1 = PCM
	binary.LittleEndian.PutUint16(header[22:24], uint16(numChannels))
	binary.LittleEndian.PutUint32(header[24:28], uint32(sampleRate))

	byteRate := sampleRate * numChannels * bitsPerSample / 8
	binary.LittleEndian.PutUint32(header[28:32], uint32(byteRate))

	blockAlign := numChannels * bitsPerSample / 8
	binary.LittleEndian.PutUint16(header[32:34], uint16(blockAlign))
	binary.LittleEndian.PutUint16(header[34:36], uint16(bitsPerSample))

	// "data" sub-chunk
	copy(header[36:40], "data")
	binary.LittleEndian.PutUint32(header[40:44], uint32(len(buf)))

	return append(header, buf...)
}

// GeminiLiveProvider implements the Provider interface using the Gemini Live BidiGenerateContent WebSocket protocol
type GeminiLiveProvider struct {
	BaseProvider
	DefaultModel       string
	ResponseModalities []string
	OutputDir          string
	WSURL              string
}

// NewGeminiLiveProvider creates a new Gemini Live provider instance
func NewGeminiLiveProvider(apiKeys []string) *GeminiLiveProvider {
	return &GeminiLiveProvider{
		BaseProvider:       BaseProvider{APIKeys: apiKeys},
		DefaultModel:       "gemini-3.1-flash-live-preview",
		ResponseModalities: []string{"AUDIO"},
		OutputDir:          "generated",
	}
}

// ListModels returns the models supported by Gemini Live
func (p *GeminiLiveProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	return SupportedLiveModels, nil
}

// FormatHistory formats past conversation messages into a compact string representation,
// omitting binary blobs and thoughts to avoid cluttering the live context.
func FormatHistory(messages []Message) string {
	if len(messages) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("\n\n=========================================\n=== CONVERSATION HISTORY ===\n=========================================\n")
	for _, msg := range messages {
		role := "User"
		if msg.Role == "model" || msg.Role == "assistant" {
			role = "Assistant"
		} else if msg.Role == "tool" || msg.Role == "function" {
			role = "Tool Result"
			name := msg.Name
			if name == "" && msg.ToolCallID != "" {
				for _, prev := range messages {
					for _, tc := range prev.ToolCalls {
						if tc.ID == msg.ToolCallID {
							name = tc.Function.Name
							break
						}
					}
					if name != "" {
						break
					}
				}
			}
			if name != "" {
				role = fmt.Sprintf("Tool Result (%s)", name)
			}
		}
		var msgText strings.Builder
		for _, part := range msg.Content {
			if part.Text != "" {
				msgText.WriteString(part.Text)
			} else if len(part.Data) > 0 || part.FileURI != "" {
				mime := part.MimeType
				if mime == "" {
					mime = "media"
				}
				msgText.WriteString(fmt.Sprintf("\n[Inline %s data provided]\n", mime))
			}
		}
		for _, tc := range msg.ToolCalls {
			msgText.WriteString(fmt.Sprintf("\n[Called Tool: %s with arguments: %s]\n", tc.Function.Name, tc.Function.Arguments))
		}
		sb.WriteString(fmt.Sprintf("%s: %s\n", role, msgText.String()))
	}
	return sb.String()
}

type liveSetupMessage struct {
	Setup liveSetup `json:"setup"`
}

type liveSetup struct {
	Model                    string                `json:"model"`
	GenerationConfig         *liveGenerationConfig `json:"generationConfig,omitempty"`
	SystemInstruction        *geminiContent        `json:"systemInstruction,omitempty"`
	OutputAudioTranscription map[string]any        `json:"outputAudioTranscription,omitempty"`
	Tools                    []liveTool            `json:"tools,omitempty"`
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
	Turns        []liveTurn `json:"turns"`
	TurnComplete bool       `json:"turnComplete"`
}

type liveTurn struct {
	Role  string            `json:"role"`
	Parts []liveContentPart `json:"parts"`
}

type liveContentPart struct {
	Text       string        `json:"text,omitempty"`
	InlineData *geminiInline `json:"inlineData,omitempty"`
}

type liveToolResponseMessage struct {
	ToolResponse liveToolResponse `json:"toolResponse"`
}

type liveToolResponse struct {
	FunctionResponses []liveFunctionResponse `json:"functionResponses"`
}

type liveFunctionResponse struct {
	ID       string         `json:"id,omitempty"`
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

type liveIncomingFrame struct {
	SetupComplete any `json:"setupComplete,omitempty"`
	ServerContent *struct {
		ModelTurn *struct {
			Parts []struct {
				Text         string              `json:"text,omitempty"`
				Thought      bool                `json:"thought,omitempty"`
				InlineData   *geminiInline       `json:"inlineData,omitempty"`
				FunctionCall *geminiFunctionCall `json:"functionCall,omitempty"`
			} `json:"parts,omitempty"`
		} `json:"modelTurn,omitempty"`
		OutputTranscription *struct {
			Text string `json:"text,omitempty"`
		} `json:"outputTranscription,omitempty"`
		TurnComplete      bool   `json:"turnComplete,omitempty"`
		Interrupted       bool   `json:"interrupted,omitempty"`
		InteractionStatus string `json:"interactionStatus,omitempty"`
	} `json:"serverContent,omitempty"`
	ToolCall *struct {
		FunctionCalls []struct {
			ID   string         `json:"id,omitempty"`
			Name string         `json:"name,omitempty"`
			Args map[string]any `json:"args,omitempty"`
		} `json:"functionCalls,omitempty"`
	} `json:"toolCall,omitempty"`
}

// Chat executes a bidirectional streaming session with Gemini Live via WebSockets
func (p *GeminiLiveProvider) Chat(ctx context.Context, req *CompletionRequest, onChunk func(*CompletionResponse)) (*CompletionResponse, error) {
	apiKey := p.GetNextKey()
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("API key is required for Gemini Live connection")
	}

	model := req.Model
	if model == "" {
		model = p.DefaultModel
	}
	if model == "" {
		model = "gemini-3.1-flash-live-preview"
	}

	// Model normalization & thinking level detection
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

	// Map gemini-3.8-live alias to models/gemini-3.8-live-extended-thinking
	if model == "gemini-3.8-live" || model == "models/gemini-3.8-live" {
		model = "models/gemini-3.8-live-extended-thinking"
	}
	// Google's 3.8 live backend currently fails with a system error during function calling.
	// Automatically route to 3.1-flash-live-preview when tools are present for reliable execution.
	if len(req.Tools) > 0 && strings.Contains(strings.ToLower(model), "3.8") {
		log.Printf("[Gemini Live] Model %s encounters server-side error during live tool calling on Google backend. Routing to models/gemini-3.1-flash-live-preview", model)
		model = "models/gemini-3.1-flash-live-preview"
	}
	if !strings.HasPrefix(model, "models/") {
		model = "models/" + model
	}

	wsURL := p.WSURL
	if wsURL == "" {
		wsURL = fmt.Sprintf("wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key=%s", url.QueryEscape(apiKey))
	}

	log.Printf("[Gemini Live] Connecting to WebSocket (Key length: %d)", len(apiKey))
	log.Printf("[Gemini Live] Targeted model formatted parameter: %s", model)

	dialer := websocket.DefaultDialer
	conn, resp, err := dialer.DialContext(ctx, wsURL, nil)
	if err != nil {
		if resp != nil {
			return nil, fmt.Errorf("gemini live ws dial failed with status %d: %w", resp.StatusCode, err)
		}
		return nil, fmt.Errorf("gemini live ws dial failed: %w", err)
	}
	defer conn.Close()

	log.Printf("[Gemini Live] Connection opened successfully.")

	// Separate system instructions and conversation messages
	var systemInstructions []string
	if req.SystemInstruction != "" {
		systemInstructions = append(systemInstructions, req.SystemInstruction)
	}

	var conversationMessages []Message
	for _, msg := range req.Messages {
		if msg.Role == "system" || msg.Role == "developer" {
			for _, part := range msg.Content {
				if part.Text != "" {
					systemInstructions = append(systemInstructions, part.Text)
				}
			}
		} else {
			conversationMessages = append(conversationMessages, msg)
		}
	}

	// Prepare history and current prompt from conversation messages only
	var historyMessages []Message
	var currentMessage *Message
	if len(conversationMessages) > 0 {
		historyMessages = conversationMessages[:len(conversationMessages)-1]
		currentMessage = &conversationMessages[len(conversationMessages)-1]
	}

	historyString := FormatHistory(historyMessages)

	var liveTurns []liveTurn
	if currentMessage != nil {
		var parts []liveContentPart
		var textParts []string
		if historyString != "" {
			textParts = append(textParts, historyString+"\n\n=== CURRENT PROMPT ===\n")
		}

		if currentMessage.Role == "tool" || currentMessage.Role == "function" {
			name := currentMessage.Name
			if name == "" && currentMessage.ToolCallID != "" {
				for _, hmsg := range historyMessages {
					for _, tc := range hmsg.ToolCalls {
						if tc.ID == currentMessage.ToolCallID {
							name = tc.Function.Name
							break
						}
					}
					if name != "" {
						break
					}
				}
			}
			if name == "" {
				name = "tool"
			}
			textParts = append(textParts, fmt.Sprintf("[Tool Result for %s]:\n", name))
		}

		for _, part := range currentMessage.Content {
			if part.Text != "" {
				textParts = append(textParts, part.Text)
			} else if len(part.Data) > 0 {
				mime := part.MimeType
				if mime == "" {
					mime = "application/octet-stream"
				}
				parts = append(parts, liveContentPart{
					InlineData: &geminiInline{
						MimeType: mime,
						Data:     part.Data,
					},
				})
			}
		}

		for _, tc := range currentMessage.ToolCalls {
			textParts = append(textParts, fmt.Sprintf("\n[Called Tool: %s with arguments: %s]\n", tc.Function.Name, tc.Function.Arguments))
		}

		if len(textParts) > 0 {
			parts = append([]liveContentPart{{Text: strings.Join(textParts, "")}}, parts...)
		}

		if len(parts) > 0 {
			role := currentMessage.Role
			if role == "assistant" {
				role = "model"
			} else if role != "model" {
				role = "user"
			}
			liveTurns = append(liveTurns, liveTurn{
				Role:  role,
				Parts: parts,
			})
		}
	}

	// Determine response modalities
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
		return nil, fmt.Errorf("failed to marshal setup frame: %w", err)
	}

	log.Printf("[Gemini Live] Outgoing setup frame: %s", string(setupJSON))
	if err := conn.WriteMessage(websocket.TextMessage, setupJSON); err != nil {
		return nil, fmt.Errorf("failed to send setup frame: %w", err)
	}

	// WAV audio accumulation and media generation commented out as requested
	/*
	var (
		audioChunks   [][]byte
		audioMimeType string
		isWavSaved    bool
		audioMutex    sync.Mutex
	)

	outDir := p.OutputDir
	if outDir == "" {
		outDir = "generated"
	}

	saveAccumulatedAudio := func() {
		audioMutex.Lock()
		defer audioMutex.Unlock()
		if len(audioChunks) == 0 || isWavSaved {
			return
		}
		isWavSaved = true

		totalLen := 0
		for _, c := range audioChunks {
			totalLen += len(c)
		}
		rawPcm := make([]byte, 0, totalLen)
		for _, c := range audioChunks {
			rawPcm = append(rawPcm, c...)
		}
		audioChunks = nil

		sampleRate := 24000
		if audioMimeType != "" {
			re := regexp.MustCompile(`rate=(\d+)`)
			matches := re.FindStringSubmatch(audioMimeType)
			if len(matches) > 1 {
				if r, err := strconv.Atoi(matches[1]); err == nil && r > 0 {
					sampleRate = r
				}
			}
		}

		wavBytes := WriteWavHeader(rawPcm, sampleRate, 1, 16)
		if err := os.MkdirAll(outDir, 0755); err != nil {
			log.Printf("[Gemini Live] Error creating generated media dir %s: %v", outDir, err)
			return
		}

		fileName := fmt.Sprintf("audio_%d_%d.wav", time.Now().UnixMilli(), rand.Intn(1000))
		filePath := filepath.Join(outDir, fileName)
		if err := os.WriteFile(filePath, wavBytes, 0644); err != nil {
			log.Printf("[Gemini Live] Error writing audio file %s: %v", filePath, err)
			return
		}
		log.Printf("[Gemini Live] Saved generated audio: %s (%d bytes)", filePath, len(wavBytes))
	}

	saveGeneratedMediaImmediately := func(inline *geminiInline) {
		if inline == nil || len(inline.Data) == 0 {
			return
		}
		mime := inline.MimeType
		if mime == "" {
			mime = "application/octet-stream"
		}
		ext := "bin"
		parts := strings.Split(mime, "/")
		if len(parts) == 2 {
			ext = parts[1]
		}
		typePrefix := "media"
		if len(parts) > 0 {
			typePrefix = parts[0]
		}

		if err := os.MkdirAll(outDir, 0755); err != nil {
			log.Printf("[Gemini Live] Error creating dir %s: %v", outDir, err)
			return
		}
		fileName := fmt.Sprintf("%s_%d_%d.%s", typePrefix, time.Now().UnixMilli(), rand.Intn(1000), ext)
		filePath := filepath.Join(outDir, fileName)
		if err := os.WriteFile(filePath, inline.Data, 0644); err != nil {
			log.Printf("[Gemini Live] Error writing generated media %s: %v", filePath, err)
			return
		}
		log.Printf("[Gemini Live] Saved generated media: %s", filePath)
	}
	*/

	var (
		fullResponse CompletionResponse
		doneChan     = make(chan struct{})
		errChan      = make(chan error, 1)
	)

	// Close WebSocket on context cancellation
	go func() {
		select {
		case <-ctx.Done():
			log.Printf("[Gemini Live] Context cancelled, closing socket.")
			conn.Close()
		case <-doneChan:
		}
	}()

	go func() {
		defer close(doneChan)
		for {
			_, msgBytes, err := conn.ReadMessage()
			if err != nil {
				if websocket.IsCloseError(err, websocket.CloseNormalClosure) || ctx.Err() != nil {
					return
				}
				log.Printf("[Gemini Live] Read error: %v", err)
				errChan <- err
				return
			}

			var frame liveIncomingFrame
			if err := json.Unmarshal(msgBytes, &frame); err != nil {
				log.Printf("[Gemini Live] Error unmarshaling incoming frame: %v", err)
				continue
			}

			// Handle setupComplete
			if frame.SetupComplete != nil {
				log.Printf("[Gemini Live] setupComplete received.")
				if len(liveTurns) > 0 {
					clientContentMsg := liveClientContentMessage{
						ClientContent: liveClientContent{
							Turns:        liveTurns,
							TurnComplete: true,
						},
					}
					clientContentJSON, _ := json.Marshal(clientContentMsg)
					log.Printf("[Gemini Live] Outgoing clientContent turns frame: %s", string(clientContentJSON))
					if err := conn.WriteMessage(websocket.TextMessage, clientContentJSON); err != nil {
						errChan <- fmt.Errorf("failed to send clientContent frame: %w", err)
						return
					}
				}
				continue
			}

			// Handle serverContent parts
			if frame.ServerContent != nil {
				if frame.ServerContent.ModelTurn != nil {
					for _, part := range frame.ServerContent.ModelTurn.Parts {
						if part.Thought && part.Text != "" {
							fullResponse.Thought += part.Text
							if onChunk != nil {
								onChunk(&CompletionResponse{Thought: part.Text})
							}
						} else if part.FunctionCall != nil {
							callID := part.FunctionCall.ID
							if callID == "" {
								callID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), len(fullResponse.ToolCalls))
							}
							argsBytes, _ := json.Marshal(part.FunctionCall.Args)
							tc := ToolCall{
								ID:   callID,
								Type: "function",
								Function: FunctionCall{
									Name:      part.FunctionCall.Name,
									Arguments: string(argsBytes),
								},
							}
							fullResponse.ToolCalls = append(fullResponse.ToolCalls, tc)
							if onChunk != nil {
								onChunk(&CompletionResponse{ToolCalls: []ToolCall{tc}})
							}
						} else if part.Text != "" {
							fullResponse.Content += part.Text
							if onChunk != nil {
								onChunk(&CompletionResponse{Content: part.Text})
							}
						} else if part.InlineData != nil {
							// Inline media / audio WAV generation commented out as requested
							/*
							if strings.HasPrefix(part.InlineData.MimeType, "audio/pcm") {
								audioMutex.Lock()
								audioChunks = append(audioChunks, part.InlineData.Data)
								audioMimeType = part.InlineData.MimeType
								audioMutex.Unlock()
							} else {
								go saveGeneratedMediaImmediately(part.InlineData)
							}
							*/
						}
					}
				}

				// Handle output transcriptions (text output corresponding to audio response)
				if frame.ServerContent.OutputTranscription != nil && frame.ServerContent.OutputTranscription.Text != "" {
					txt := frame.ServerContent.OutputTranscription.Text
					fullResponse.Content += txt
					if onChunk != nil {
						onChunk(&CompletionResponse{Content: txt})
					}
				}

				if frame.ServerContent.TurnComplete {
					log.Printf("[Gemini Live] Server turnComplete flag received. interactionStatus=%s, toolCalls=%d",
						frame.ServerContent.InteractionStatus, len(fullResponse.ToolCalls))
					if len(fullResponse.ToolCalls) > 0 {
						return
					}
					if frame.ServerContent.InteractionStatus == "IN_PROGRESS" {
						continue
					}
					return
				}
			}

			// Handle tool call events
			if frame.ToolCall != nil && len(frame.ToolCall.FunctionCalls) > 0 {
				log.Printf("[Gemini Live] Server function call request detected: %d calls", len(frame.ToolCall.FunctionCalls))
				for _, fc := range frame.ToolCall.FunctionCalls {
					callID := fc.ID
					if callID == "" {
						callID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), len(fullResponse.ToolCalls))
					}
					argsBytes, _ := json.Marshal(fc.Args)
					tc := ToolCall{
						ID:   callID,
						Type: "function",
						Function: FunctionCall{
							Name:      fc.Name,
							Arguments: string(argsBytes),
						},
					}
					fullResponse.ToolCalls = append(fullResponse.ToolCalls, tc)
					if onChunk != nil {
						onChunk(&CompletionResponse{ToolCalls: []ToolCall{tc}})
					}
				}
				// Return immediately so the tool call is handed back to the caller (Copilot)
				return
			}
		}
	}()

	select {
	case err := <-errChan:
		// saveAccumulatedAudio()
		return nil, err
	case <-doneChan:
		// saveAccumulatedAudio()
		return &fullResponse, nil
	case <-ctx.Done():
		// saveAccumulatedAudio()
		return nil, ctx.Err()
	}
}

// SendAudio sends real-time PCM audio chunks (16kHz) via an active WebSocket connection
func (p *GeminiLiveProvider) SendAudio(conn *websocket.Conn, pcmData []byte, sampleRate int) error {
	if sampleRate <= 0 {
		sampleRate = 16000
	}
	payload := map[string]any{
		"realtimeInput": map[string]any{
			"mediaChunks": []map[string]any{
				{
					"mimeType": fmt.Sprintf("audio/pcm;rate=%d", sampleRate),
					"data":     base64.StdEncoding.EncodeToString(pcmData),
				},
			},
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, data)
}

// SendVideoFrame sends a JPEG or PNG video frame via an active WebSocket connection
func (p *GeminiLiveProvider) SendVideoFrame(conn *websocket.Conn, imgData []byte, mimeType string) error {
	if mimeType == "" {
		mimeType = "image/jpeg"
	}
	payload := map[string]any{
		"realtimeInput": map[string]any{
			"mediaChunks": []map[string]any{
				{
					"mimeType": mimeType,
					"data":     base64.StdEncoding.EncodeToString(imgData),
				},
			},
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, data)
}
