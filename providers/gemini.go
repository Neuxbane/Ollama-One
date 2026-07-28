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
	"regexp"
	"strings"
	"time"
)

type GeminiProvider struct {
	BaseProvider
	BaseURL string
}

func (p *GeminiProvider) getBaseURL() string {
	if p.BaseURL != "" {
		return strings.TrimSuffix(p.BaseURL, "/")
	}
	return "https://generativelanguage.googleapis.com"
}

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
	Data     []byte `json:"data"` // This will be base64 encoded by json.Marshal
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

type geminiRequest struct {
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	Contents          []geminiContent         `json:"contents"`
	Tools             []geminiTool            `json:"tools,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
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

func (p *GeminiProvider) ListModels(ctx context.Context) ([]ModelInfo, error) {
	key := p.GetNextKey()
	if key == "" {
		log.Printf("[Gemini] ListModels error: no API key found")
		return nil, fmt.Errorf("no gemini api key configured")
	}
	var models []ModelInfo
	pageToken := ""

	page := 1
	for {
		url := fmt.Sprintf("%s/v1beta/models?key=%s", p.getBaseURL(), key)
		if pageToken != "" {
			url += fmt.Sprintf("&pageToken=%s", pageToken)
		}

		req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
		if err != nil {
			log.Printf("[Gemini] ListModels page %d request error: %v", page, err)
			return nil, err
		}

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			log.Printf("[Gemini] ListModels page %d HTTP error: %v", page, err)
			return nil, err
		}

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			log.Printf("[Gemini] ListModels page %d API error (status %d): %s", page, resp.StatusCode, string(body))
			return nil, fmt.Errorf("gemini api error (status %d): %s", resp.StatusCode, string(body))
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
			log.Printf("[Gemini] ListModels page %d JSON decode error: %v", page, err)
			return nil, err
		}
		resp.Body.Close()

		log.Printf("[Gemini] ListModels page %d fetched %d models from API", page, len(data.Models))

		for _, m := range data.Models {
			id := strings.TrimPrefix(m.Name, "models/")
			caps := []string{"vision", "tools", "chat", "completion"}
			if m.Thinking {
				caps = append(caps, "thinking")
			}

			models = append(models, ModelInfo{
				ID:           id,
				Name:         m.DisplayName,
				ContextSize:  170000,
				Capabilities: caps,
			})
		}

		if data.NextPageToken == "" {
			break
		}
		pageToken = data.NextPageToken
		page++
	}

	log.Printf("[Gemini] ListModels completed: %d generateContent models total", len(models))
	return models, nil
}

func (p *GeminiProvider) Chat(ctx context.Context, req *CompletionRequest, onChunk func(*CompletionResponse)) (*CompletionResponse, error) {
	key := p.GetNextKey()
	modelLower := strings.ToLower(req.Model)

	if strings.Contains(modelLower, "antigravity") {
		return p.handleAntigravityInteraction(ctx, key, req, onChunk)
	}
	if strings.Contains(modelLower, "live") {
		return p.handleLiveModel(ctx, key, req, onChunk)
	}

	// Prepare request body
	gemReq := geminiRequest{
		Contents: make([]geminiContent, len(req.Messages)),
	}

	if req.SystemInstruction != "" {
		gemReq.SystemInstruction = &geminiContent{
			Parts: []geminiPart{{Text: req.SystemInstruction}},
		}
	}

	for i, msg := range req.Messages {
		role := msg.Role
		switch role {
		case "assistant":
			role = "model"
		case "system", "developer", "":
			role = "user"
		case "tool":
			role = "function" // Gemini uses 'function' role for tool results (often mapped to 'user' in some SDKs, but let's handle parts carefully)
		}
		
		gemReq.Contents[i] = geminiContent{
			Role: role,
		}

		// Handle ToolCalls from previous assistant messages
		for _, tc := range msg.ToolCalls {
			var args map[string]any
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				args = make(map[string]any)
			}
			gemReq.Contents[i].Parts = append(gemReq.Contents[i].Parts, geminiPart{
				FunctionCall: &geminiFunctionCall{
					Name: tc.Function.Name,
					Args: args,
					ID:   tc.ID,
				},
			})
		}

		for _, part := range msg.Content {
			p := geminiPart{}
			switch part.Type {
			case ContentTypeText:
				if role == "function" {
					// This is a tool result. Map it to FunctionResponse.
					// We need to find the tool call ID. 
					// For Ollama/OpenAI, it's often in the message content or a separate field.
					// Assuming the tool call name matches the function name.
					p.FunctionResponse = &geminiFunctionResponse{
						Name: "unknown", // Will be fixed below if possible
						Response: map[string]any{
							"result": part.Text,
						},
					}
					// Try to parse as JSON if it looks like one
					var jsonResult any
					if err := json.Unmarshal([]byte(part.Text), &jsonResult); err == nil {
						p.FunctionResponse.Response = map[string]any{
							"result": jsonResult,
						}
					}
				} else {
					p.Text = part.Text
				}
			case ContentTypeImage, ContentTypeDocument:
				if part.FileURI != "" {
					p.FileData = &geminiFileData{
						MimeType: part.MimeType,
						FileURI:  part.FileURI,
					}
				} else {
					p.InlineData = &geminiInline{
						MimeType: part.MimeType,
						Data:     part.Data,
					}
				}
			}
			if p.Text != "" || p.InlineData != nil || p.FileData != nil || p.FunctionResponse != nil {
				gemReq.Contents[i].Parts = append(gemReq.Contents[i].Parts, p)
			}
		}

		// If it's a tool result, we need to ensure the Name is correct.
		// In a session-based approach, we'd look up the last tool call.
		if role == "function" {
			// Find the last model message with a tool call
			for j := i - 1; j >= 0; j-- {
				if gemReq.Contents[j].Role == "model" {
					for _, p := range gemReq.Contents[j].Parts {
						if p.FunctionCall != nil {
							for k := range gemReq.Contents[i].Parts {
								if gemReq.Contents[i].Parts[k].FunctionResponse != nil {
									gemReq.Contents[i].Parts[k].FunctionResponse.Name = p.FunctionCall.Name
								}
							}
						}
					}
					break
				}
			}
		}
	}

	for _, tool := range req.Tools {
		if tool.GoogleSearch {
			gemReq.Tools = append(gemReq.Tools, geminiTool{
				GoogleSearch: make(map[string]any),
			})
		}
		if len(tool.Functions) > 0 {
			var funcs []geminiFunction
			for _, f := range tool.Functions {
				funcs = append(funcs, geminiFunction{
					Name:        f.Name,
					Description: f.Description,
					Parameters:  sanitizeGeminiSchemaMap(f.Parameters),
				})
			}
			if len(funcs) > 0 {
				gemReq.Tools = append(gemReq.Tools, geminiTool{FunctionDeclarations: funcs})
			}
		}
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

	// Always use streaming if possible
	url := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse&key=%s", p.getBaseURL(), req.Model, key)
	
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
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
		return nil, fmt.Errorf("gemini api error: %s", string(body))
	}

	fullContent := ""
	var allToolCalls []ToolCall
	reader := bufio.NewReader(resp.Body)
	
	inThought := false
	pendingText := "" // Buffer for potentially intercepted text

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
				// Handle standard text
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
						
						// --- REAL-TIME TOOL INTERCEPTION ---
						combinedText := pendingText + textToYield
						
						// Look for opening tags/patterns that suggest a tool call is starting
						tags := []string{"<tool_code>", "<tool_call>", "<toolUse>", "<tool_use>", "<function_calls>", "```json", "call:"}
						foundTag := false
						tagIndex := -1
						
						for _, tag := range tags {
							if idx := strings.Index(combinedText, tag); idx != -1 {
								foundTag = true
								if tagIndex == -1 || idx < tagIndex {
									tagIndex = idx
								}
							}
						}
						
						if foundTag {
							// Yield everything BEFORE the tag
							toYield := combinedText[:tagIndex]
							if toYield != "" {
								fullContent += toYield
								if onChunk != nil {
									onChunk(&CompletionResponse{Content: toYield})
								}
							}
							// Keep the tag and everything after it in the buffer
							pendingText = combinedText[tagIndex:]
							
							// Check if the buffer now contains a COMPLETE block
							closers := map[string]string{
								"<tool_code>":      "</tool_code>",
								"<tool_call>":      "</tool_call>",
								"<toolUse>":        "</toolUse>",
								"<tool_use>":       "</tool_use>",
								"<function_calls>": "</function_calls>",
								"```json":          "```",
								"call:":            "}",
							}
							
							for opener, closer := range closers {
								if strings.HasPrefix(pendingText, opener) {
									if cIdx := strings.Index(pendingText[len(opener):], closer); cIdx != -1 {
										// Found the end!
										endIdx := len(opener) + cIdx + len(closer)
										_ = pendingText[:endIdx] // fullBlock
										
										// Extract the JSON content
										rawArgs := pendingText[len(opener) : len(opener)+cIdx]
										rawArgs = strings.TrimSpace(rawArgs)
										
										// Try to parse it
										var toolItems []map[string]any
										if err := json.Unmarshal([]byte(rawArgs), &toolItems); err != nil {
											// Not an array, try single object
											var singleItem map[string]any
											// Fix common unquoted issues
											fixArgs := rawArgs
											keyFixer := regexp.MustCompile(`([{,])\s*([a-zA-Z0-9_]+)\s*:`)
											fixArgs = keyFixer.ReplaceAllString(fixArgs, `$1"$2":`)
											if err := json.Unmarshal([]byte(fixArgs), &singleItem); err == nil {
												toolItems = append(toolItems, singleItem)
											}
										}
										
										for _, item := range toolItems {
											funcName := opener
											if opener == "call:" {
												// In call:name{args}, name is before {
												re := regexp.MustCompile(`^([a-zA-Z0-9_]+)\s*\{`)
												if m := re.FindStringSubmatch(rawArgs); len(m) >= 2 {
													funcName = m[1]
												}
											} else {
												if n, ok := item["name"].(string); ok {
													funcName = n
				} else if t, ok := item["tool"].(string); ok {
													funcName = t
												}
											}
											
											// Clean up function name
											prefixes := []string{"tool_use:", "tool_code:", "tool_call:", "call:"}
											for _, pfx := range prefixes {
												funcName = strings.TrimPrefix(funcName, pfx)
											}
											funcName = strings.ReplaceAll(funcName, " ", "_")
											args := item
											if a, ok := item["arguments"].(map[string]any); ok {
												args = a
											} else if a, ok := item["args"].(map[string]any); ok {
												args = a
											}

											fixedArgs, _ := p.fixToolCall(funcName, args, req.Tools)
											argsBytes, _ := json.Marshal(fixedArgs)
											
											toolCall := ToolCall{
												ID:   fmt.Sprintf("call_stream_%d", len(allToolCalls)),
												Type: "function",
												Function: FunctionCall{
													Name:      funcName,
													Arguments: string(argsBytes),
												},
											}
											allToolCalls = append(allToolCalls, toolCall)
											if onChunk != nil {
												onChunk(&CompletionResponse{ToolCalls: []ToolCall{toolCall}})
											}
										}
										
										// Clear the consumed block from buffer
										pendingText = pendingText[endIdx:]
									}
									break
								}
							}
						} else {
							// No tag found, yield everything and clear buffer
							fullContent += combinedText
							if onChunk != nil {
								onChunk(&CompletionResponse{Content: combinedText})
							}
							pendingText = ""
						}
					}
				}
				// Handle native function call logic
				if part.FunctionCall != nil {
					if inThought {
						closing := "\n</think>\n\n"
						fullContent += closing
						if onChunk != nil {
							onChunk(&CompletionResponse{Content: closing})
						}
						inThought = false
					}

					// Clean up function name
					funcName := part.FunctionCall.Name
					prefixes := []string{"tool_use:", "tool_code:", "tool_call:", "call:"}
					for _, pfx := range prefixes {
						funcName = strings.TrimPrefix(funcName, pfx)
					}

					// Generic correction for tool calls
					fixedArgs, _ := p.fixToolCall(funcName, part.FunctionCall.Args, req.Tools)
					argsBytes, _ := json.Marshal(fixedArgs)
					
					log.Printf("FIXED Standard Tool Call: %s(%s)", funcName, string(argsBytes))

					toolCall := ToolCall{
						ID:   part.FunctionCall.ID,
						Type: "function",
						Function: FunctionCall{
							Name:      funcName,
							Arguments: string(argsBytes),
						},
					}
					allToolCalls = append(allToolCalls, toolCall)

					if onChunk != nil {
						onChunk(&CompletionResponse{ToolCalls: []ToolCall{toolCall}})
					}
				}
			}

			// Fallback for malformed function calls (common in Gemma models)
			if cand.FinishReason == "MALFORMED_FUNCTION_CALL" && cand.FinishMessage != "" {
				log.Printf("Handling malformed function call: %s", cand.FinishMessage)
				
				// 1. Extract function name and raw arguments string
				// This regex skips any prefix noise and looks for "name{" or "call:name{"
				re := regexp.MustCompile(`(?s).*?(?:call:)?([a-zA-Z0-9_]+)[\s\n]*\{+(.*)`)
				matches := re.FindStringSubmatch(cand.FinishMessage)
				if len(matches) >= 3 {
					funcName := matches[1]
					argsStr := strings.TrimSpace(matches[2])
					
					// 2. Clean up the arguments string (remove all trailing noise/braces)
					// We only want the content inside the outermost braces
					lastBrace := strings.LastIndex(argsStr, "}")
					if lastBrace != -1 {
						argsStr = argsStr[:lastBrace]
					}
					argsStr = strings.TrimSpace(argsStr)
					
					// 3. Ensure it starts with { and ends with }
					if !strings.HasPrefix(argsStr, "{") {
						argsStr = "{" + argsStr + "}"
					}
					
					// 4. Fix unquoted keys
					keyFixer := regexp.MustCompile(`([{,])\s*([a-zA-Z0-9_]+)\s*:`)
					argsStr = keyFixer.ReplaceAllString(argsStr, `$1"$2":`)
					
					// 5. Fix single quotes (convert 'string' to "string")
					quoteFixer := regexp.MustCompile(`'([^']*)'`)
					argsStr = quoteFixer.ReplaceAllString(argsStr, `"$1"`)
					
					var args map[string]any
					if err := json.Unmarshal([]byte(argsStr), &args); err == nil {
						fixedArgs, _ := p.fixToolCall(funcName, args, req.Tools)
						finalArgs, _ := json.Marshal(fixedArgs)
						
						log.Printf("PASSED THROUGH Tool Call: %s(%s)", funcName, string(finalArgs))

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
					} else {
						log.Printf("Failed to parse malformed args JSON: %v (Raw: %s)", err, argsStr)
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

	// Final check: scan fullContent for raw JSON tool calls or <tool_code>/<tool_call> blocks
	// (Some models output these when they are confused by the API)
	if fullContent != "" && len(allToolCalls) == 0 {
		// ...
		// Look for <tool_code>{...}</tool_code>, <tool_call>{...}</tool_call> or just ```json\n{...}\n```
		// Now supports both single objects {} and arrays []
		reToolCode := regexp.MustCompile(`(?s)<tool_code>\s*([\{\[].*?[\}\]])\s*</tool_code>`)
		reToolCall := regexp.MustCompile(`(?s)<tool_call>\s*([\{\[].*?[\}\]])\s*</tool_call>`)
		reJSONBlock := regexp.MustCompile("(?s)```json\\s*([{\\[].*?[}\\]])\\s*```")
		
		var rawArgs string
		var matchStr string
		
		if m := reToolCode.FindStringSubmatch(fullContent); len(m) >= 2 {
			rawArgs = m[1]
			matchStr = m[0]
		} else if m := reToolCall.FindStringSubmatch(fullContent); len(m) >= 2 {
			rawArgs = m[1]
			matchStr = m[0]
		} else if m := reJSONBlock.FindStringSubmatch(fullContent); len(m) >= 2 {
			rawArgs = m[1]
			matchStr = m[0]
		}
		
		if rawArgs != "" {
			// Fix unquoted keys and values in raw output
			keyFixer := regexp.MustCompile(`([{,])\s*([a-zA-Z0-9_]+)\s*:`)
			rawArgs = keyFixer.ReplaceAllString(rawArgs, `$1"$2":`)
			
			// Fix unquoted string values (avoid numbers/booleans/null)
			// This specifically targets unquoted paths and strings
			valueFixer := regexp.MustCompile(`:\s*([^"\{\}\[\]\s,0-9tfn\-\.][^\{\}\[\]\s,]*)\s*([\},])`)
			rawArgs = valueFixer.ReplaceAllString(rawArgs, `:"$1"$2`)

			// Try to parse as array first, then as single object
			var toolItems []map[string]any
			if err := json.Unmarshal([]byte(rawArgs), &toolItems); err != nil {
				// Not an array, try single object
				var singleItem map[string]any
				if err := json.Unmarshal([]byte(rawArgs), &singleItem); err == nil {
					toolItems = append(toolItems, singleItem)
				}
			}

			for _, item := range toolItems {
				funcName := "read_file" // Default
				args := item
				
				// Extract tool name if present in JSON (e.g. {"tool": "read_file", "arguments": {...}})
				if t, ok := item["tool"].(string); ok {
					funcName = t
					if a, ok := item["arguments"].(map[string]any); ok {
						args = a
					}
				} else if n, ok := item["name"].(string); ok {
					funcName = n
					if a, ok := item["arguments"].(map[string]any); ok {
						args = a
					}
				}
				
				fixedArgs, _ := p.fixToolCall(funcName, args, req.Tools)
				finalArgs, _ := json.Marshal(fixedArgs)
				
				log.Printf("FIXED Raw Tool Call: %s(%s)", funcName, string(finalArgs))
				
				toolCall := ToolCall{
					ID:   fmt.Sprintf("call_raw_%d", len(allToolCalls)),
					Type: "function",
					Function: FunctionCall{
						Name:      funcName,
						Arguments: string(finalArgs),
					},
				}
				allToolCalls = append(allToolCalls, toolCall)
			}
			
			if len(allToolCalls) > 0 {
				// Strip the tool calls from content so VSCode triggers them
				fullContent = strings.ReplaceAll(fullContent, matchStr, "")
				fullContent = strings.TrimSpace(fullContent)
			}
		}
	}

	return &CompletionResponse{Content: fullContent, ToolCalls: allToolCalls}, nil
}

// fixToolCall attempts to map model-provided arguments to the tool's required schema
func (p *GeminiProvider) fixToolCall(funcName string, args map[string]any, availableTools []Tool) (map[string]any, error) {
	if args == nil {
		args = make(map[string]any)
	}

	// Find the matching function definition
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
		// BYPASS: If no definition found, still allow the tool call to pass through
		return args, nil
	}

	// Extract required properties and types from the JSON schema
	if params, ok := targetFunc.Parameters["properties"].(map[string]any); ok {
		required := []string{}
		if req, ok := targetFunc.Parameters["required"].([]any); ok {
			for _, r := range req {
				if s, ok := r.(string); ok {
					required = append(required, s)
				}
			}
		}
		
		log.Printf("Tool %s required parameters: %v", funcName, required)

		// 1. Fuzzy match existing arguments to required ones
		for _, reqKey := range required {
			if _, exists := args[reqKey]; exists {
				continue
			}

			// Try to find a fuzzy match
			for argKey, argVal := range args {
				// Skip if this argKey is already a correct required key
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

				// Fuzzy match logic: check for common aliases or substrings
				lArg := strings.ToLower(argKey)
				lReq := strings.ToLower(reqKey)
				if (lArg == "path" && lReq == "filepath") || 
				   (lArg == "content" && lReq == "text") ||
				   strings.Contains(lReq, lArg) || strings.Contains(lArg, lReq) {
					log.Printf("Mapping argument %s to %s for tool %s", argKey, reqKey, funcName)
					args[reqKey] = argVal
					delete(args, argKey)
					break
				}
			}
		}

		// 2. Inject default "Zero Values" for missing required parameters based on type
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
				// Special heuristic: startLine usually wants 1, everything else 0
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
			log.Printf("Injected missing required key: %s = %v for tool %s", reqKey, args[reqKey], funcName)
		}
	}

	return args, nil
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
			// Remove schema properties that Google Gemini REST API rejects
			switch k {
			case "$comment", "$schema", "$id", "title", "enumDescriptions", "examples", "default", "additionalProperties":
				continue
			}
			cleaned[k] = sanitizeGeminiSchema(child)
		}

		// Handle type arrays e.g. ["string", "null"] -> type: "string", nullable: true
		if typeVal, exists := cleaned["type"]; exists {
			if typeSlice, ok := typeVal.([]any); ok {
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
					cleaned["type"] = firstType
				} else {
					cleaned["type"] = "string"
				}
				if isNullable {
					cleaned["nullable"] = true
				}
			}
		}

		// Validate required fields: every item in "required" MUST exist in "properties"
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

func (p *GeminiProvider) handleAntigravityInteraction(ctx context.Context, key string, req *CompletionRequest, onChunk func(*CompletionResponse)) (*CompletionResponse, error) {
	var inputBuilder strings.Builder
	for _, msg := range req.Messages {
		for _, part := range msg.Content {
			if part.Type == ContentTypeText && part.Text != "" {
				if inputBuilder.Len() > 0 {
					inputBuilder.WriteString("\n")
				}
				inputBuilder.WriteString(part.Text)
			}
		}
	}

	payload := map[string]any{
		"agent":      req.Model,
		"input":      inputBuilder.String(),
		"background": true,
		"tools": []map[string]any{
			{"type": "code_execution"},
			{"type": "google_search"},
			{"type": "url_context"},
		},
		"environment": map[string]any{
			"type":    "remote",
			"network": "disabled",
		},
	}

	bodyBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	createURL := fmt.Sprintf("%s/v1beta/interactions?key=%s", p.getBaseURL(), key)
	log.Printf("[Gemini Antigravity] Creating interaction for model %s via %s", req.Model, createURL)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", createURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini interaction api error (%d): %s", resp.StatusCode, string(body))
	}

	var createResp map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&createResp); err != nil {
		return nil, err
	}

	interactionID, _ := createResp["id"].(string)
	if interactionID == "" {
		if name, ok := createResp["name"].(string); ok {
			interactionID = strings.TrimPrefix(name, "interactions/")
		}
	}

	log.Printf("[Gemini Antigravity] Created interaction %s, starting streaming step poll...", interactionID)

	pollURL := fmt.Sprintf("%s/v1beta/interactions/%s?key=%s", p.getBaseURL(), interactionID, key)
	ticker := time.NewTicker(400 * time.Millisecond)
	defer ticker.Stop()

	var fullContent strings.Builder
	var allToolCalls []ToolCall
	lastStepProcessed := 0
	inThought := false

	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-ticker.C:
			pReq, err := http.NewRequestWithContext(ctx, "GET", pollURL, nil)
			if err != nil {
				return nil, err
			}

			pResp, err := http.DefaultClient.Do(pReq)
			if err != nil {
				return nil, err
			}

			var pData map[string]any
			err = json.NewDecoder(pResp.Body).Decode(&pData)
			pResp.Body.Close()
			if err != nil {
				continue
			}

			status, _ := pData["status"].(string)

			// Process new steps incrementally for live streaming
			if steps, ok := pData["steps"].([]any); ok {
				for i := lastStepProcessed; i < len(steps); i++ {
					stepMap, ok := steps[i].(map[string]any)
					if !ok {
						continue
					}
					stepType, _ := stepMap["type"].(string)

					switch stepType {
					case "thought":
						var thoughtText string
						if summaries, ok := stepMap["summary"].([]any); ok {
							for _, sum := range summaries {
								if sm, ok := sum.(map[string]any); ok {
									if txt, ok := sm["text"].(string); ok {
										thoughtText += txt
									}
								}
							}
						}
						if thoughtText != "" {
							if !inThought {
								header := "<think>\n"
								fullContent.WriteString(header)
								if onChunk != nil {
									onChunk(&CompletionResponse{Content: header})
								}
								inThought = true
							}
							fullContent.WriteString(thoughtText)
							if onChunk != nil {
								onChunk(&CompletionResponse{Content: thoughtText})
							}
						}

					case "model_output":
						if inThought {
							closing := "\n</think>\n\n"
							fullContent.WriteString(closing)
							if onChunk != nil {
								onChunk(&CompletionResponse{Content: closing})
							}
							inThought = false
						}

						var outputText string
						if contents, ok := stepMap["content"].([]any); ok {
							for _, c := range contents {
								if cm, ok := c.(map[string]any); ok {
									if txt, ok := cm["text"].(string); ok {
										outputText += txt
									}
								}
							}
						}
						if outputText != "" {
							fullContent.WriteString(outputText)
							if onChunk != nil {
								onChunk(&CompletionResponse{Content: outputText})
							}
						}

					case "tool_call", "function_call", "tool_use":
						if inThought {
							closing := "\n</think>\n\n"
							fullContent.WriteString(closing)
							if onChunk != nil {
								onChunk(&CompletionResponse{Content: closing})
							}
							inThought = false
						}

						funcName := ""
						var funcArgs map[string]any

						if fc, ok := stepMap["function_call"].(map[string]any); ok {
							funcName, _ = fc["name"].(string)
							if args, ok := fc["args"].(map[string]any); ok {
								funcArgs = args
							} else if args, ok := fc["arguments"].(map[string]any); ok {
								funcArgs = args
							}
						} else if tc, ok := stepMap["tool_call"].(map[string]any); ok {
							funcName, _ = tc["name"].(string)
							if args, ok := tc["args"].(map[string]any); ok {
								funcArgs = args
							}
						} else if fn, ok := stepMap["name"].(string); ok {
							funcName = fn
							if args, ok := stepMap["args"].(map[string]any); ok {
								funcArgs = args
							}
						}

						if funcName != "" {
							if funcArgs == nil {
								funcArgs = make(map[string]any)
							}
							argsBytes, _ := json.Marshal(funcArgs)
							toolCall := ToolCall{
								ID:   fmt.Sprintf("call_interaction_%d", i),
								Type: "function",
								Function: FunctionCall{
									Name:      funcName,
									Arguments: string(argsBytes),
								},
							}
							allToolCalls = append(allToolCalls, toolCall)
							log.Printf("[Gemini Antigravity] Tool Call step %d: %s(%s)", i, funcName, string(argsBytes))
							if onChunk != nil {
								onChunk(&CompletionResponse{ToolCalls: []ToolCall{toolCall}})
							}
						}
					}
					lastStepProcessed = i + 1
				}
			}

			if status == "completed" || status == "DONE" || status == "SUCCEEDED" {
				if inThought {
					closing := "\n</think>\n\n"
					fullContent.WriteString(closing)
					if onChunk != nil {
						onChunk(&CompletionResponse{Content: closing})
					}
					inThought = false
				}
				log.Printf("[Gemini Antigravity] Interaction %s completed with %d tool call(s)", interactionID, len(allToolCalls))
				return &CompletionResponse{
					Content:   fullContent.String(),
					ToolCalls: allToolCalls,
				}, nil
			} else if status == "failed" || status == "FAILED" || status == "ERROR" {
				errDetail, _ := json.Marshal(pData["error"])
				return nil, fmt.Errorf("gemini interaction failed: %s", string(errDetail))
			}
		}
	}
}

func extractInteractionOutput(data map[string]any) string {
	if outputs, ok := data["outputs"].([]any); ok {
		var sb strings.Builder
		for _, out := range outputs {
			if m, ok := out.(map[string]any); ok {
				if text, ok := m["text"].(string); ok {
					sb.WriteString(text)
				}
			}
		}
		if sb.Len() > 0 {
			return sb.String()
		}
	}
	if text, ok := data["response"].(string); ok {
		return text
	}
	return ""
}

func (p *GeminiProvider) handleLiveModel(ctx context.Context, key string, req *CompletionRequest, onChunk func(*CompletionResponse)) (*CompletionResponse, error) {
	log.Printf("[Gemini Live] Handling live model session request for %s", req.Model)

	gemReq := geminiRequest{
		Contents: make([]geminiContent, len(req.Messages)),
	}

	for i, msg := range req.Messages {
		role := msg.Role
		if role == "assistant" {
			role = "model"
		} else if role != "user" {
			role = "user"
		}
		parts := make([]geminiPart, 0)
		for _, p := range msg.Content {
			if p.Text != "" {
				parts = append(parts, geminiPart{Text: p.Text})
			}
		}
		gemReq.Contents[i] = geminiContent{Role: role, Parts: parts}
	}

	bodyBytes, err := json.Marshal(gemReq)
	if err != nil {
		return nil, err
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse&key=%s", p.getBaseURL(), req.Model, key)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(bodyBytes))
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
		return nil, fmt.Errorf("gemini live api error (%d): %s", resp.StatusCode, string(body))
	}

	fullContent := ""
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
		var gemResp geminiResponse
		if err := json.Unmarshal([]byte(data), &gemResp); err != nil {
			continue
		}

		for _, cand := range gemResp.Candidates {
			for _, part := range cand.Content.Parts {
				if part.Text != "" {
					fullContent += part.Text
					if onChunk != nil {
						onChunk(&CompletionResponse{Content: part.Text})
					}
				}
			}
		}
	}

	return &CompletionResponse{Content: fullContent}, nil
}
