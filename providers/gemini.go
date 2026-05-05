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
)

type GeminiProvider struct {
	BaseProvider
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
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models?key=%s", key)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini api error: %s", string(body))
	}

	var data struct {
		Models []struct {
			Name            string   `json:"name"`
			DisplayName     string   `json:"displayName"`
			InputTokenLimit int      `json:"inputTokenLimit"`
			Thinking        bool     `json:"thinking"`
			SupportedMethods []string `json:"supportedGenerationMethods"`
		} `json:"models"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}

	var models []ModelInfo
	for _, m := range data.Models {
		// Filter for models that support generating content
		supportsChat := false
		for _, method := range m.SupportedMethods {
			if method == "generateContent" {
				supportsChat = true
				break
			}
		}
		if !supportsChat {
			continue
		}

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

	return models, nil
}

func (p *GeminiProvider) Chat(ctx context.Context, req *CompletionRequest, onChunk func(*CompletionResponse)) (*CompletionResponse, error) {
	key := p.GetNextKey()
	
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
					Parameters:  f.Parameters,
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
	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:streamGenerateContent?alt=sse&key=%s", req.Model, key)
	
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
