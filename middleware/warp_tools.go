package middleware

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strconv"
	"strings"
	"time"

	"github.com/Neuxbane/Ollama-One/providers"
)

func init() {
	Register("warp_tools", WarpToolsModifier)
	Register("meta_tools", WarpToolsModifier)
}

// WarpToolsConfig stores the configuration and tool index for dynamic tool retrieval
type WarpToolsConfig struct {
	TopK          int
	OriginalTools []providers.Tool
	Index         *SemanticToolIndex
}

// GetWarpToolsConfig extracts WarpToolsConfig from request metadata if present
func GetWarpToolsConfig(req *providers.CompletionRequest) (*WarpToolsConfig, bool) {
	if req == nil || req.Metadata == nil {
		return nil, false
	}
	cfg, ok := req.Metadata["warp_tools"].(*WarpToolsConfig)
	return cfg, ok
}

// WarpToolsModifier handles the :warp_tools(top_k=10) model directive.
// If len(req.Tools) <= top_k, it leaves the tools intact and bypasses meta-tools.
// If len(req.Tools) > top_k, it replaces the tools with searchTools and execute meta-tools.
func WarpToolsModifier(req *providers.CompletionRequest, args []string) error {
	topK := 10 // default top_k

	for _, arg := range args {
		arg = strings.TrimSpace(arg)
		lower := strings.ToLower(arg)
		if strings.HasPrefix(lower, "top_k=") {
			val := strings.TrimSpace(arg[6:])
			if n, err := strconv.Atoi(val); err == nil && n > 0 {
				topK = n
			}
		} else if strings.HasPrefix(lower, "topk=") {
			val := strings.TrimSpace(arg[5:])
			if n, err := strconv.Atoi(val); err == nil && n > 0 {
				topK = n
			}
		} else if n, err := strconv.Atoi(arg); err == nil && n > 0 {
			topK = n
		}
	}

	// If tools count is under or equal to top_k, do not use meta-tools
	if len(req.Tools) <= topK {
		log.Printf("[MIDDLEWARE][WARP_TOOLS] Tools count (%d) <= top_k (%d); bypassing meta-tools", len(req.Tools), topK)
		return nil
	}

	log.Printf("[MIDDLEWARE][WARP_TOOLS] Tools count (%d) > top_k (%d); activating warp_tools meta-tools", len(req.Tools), topK)

	cfg := &WarpToolsConfig{
		TopK:          topK,
		OriginalTools: req.Tools,
		Index:         NewSemanticToolIndex(req.Tools),
	}

	if req.Metadata == nil {
		req.Metadata = make(map[string]any)
	}
	req.Metadata["warp_tools"] = cfg

	// Replace tools with the two meta-tools
	req.Tools = getMetaTools()

	// Append concise system instruction guidance
	guidance := "\n\n[TOOL SYSTEM INSTRUCTION]\nYou have access to a large library of tools. To execute functions efficiently:\n1. First call `searchTools(query)` with a semantic query describing the tool or capability you need.\n2. Review the matching tool schemas returned to you.\n3. Call `execute(tools=[{\"name\": \"...\", \"arguments\": {...}}])` (or call the tool directly) to invoke it."

	if req.SystemInstruction != "" {
		req.SystemInstruction += guidance
	} else {
		req.SystemInstruction = strings.TrimSpace(guidance)
	}

	return nil
}

func getMetaTools() []providers.Tool {
	return []providers.Tool{
		{
			Type:        "function",
			Name:        "searchTools",
			Description: "Semantically search available tools by query. Use this whenever you need external capabilities or tools.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "Semantic search query describing the capability, action, or tool you are searching for",
					},
				},
				"required": []string{"query"},
			},
		},
		{
			Type:        "function",
			Name:        "execute",
			Description: "Execute one or more tools that were discovered via searchTools.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"tools": map[string]any{
						"type":        "array",
						"description": "List of tool invocations to execute",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"name": map[string]any{
									"type":        "string",
									"description": "Name of the tool to execute",
								},
								"arguments": map[string]any{
									"type":        "object",
									"description": "Arguments/parameters to pass to the tool",
								},
							},
							"required": []string{"name", "arguments"},
						},
					},
				},
				"required": []string{"tools"},
			},
		},
	}
}

// ExecuteChat coordinates provider.Chat with automatic warp_tools resolution if active
func ExecuteChat(ctx context.Context, provider providers.Provider, req *providers.CompletionRequest, onChunk func(*providers.CompletionResponse)) (*providers.CompletionResponse, error) {
	cfg, ok := GetWarpToolsConfig(req)
	if !ok || cfg == nil {
		return provider.Chat(ctx, req, onChunk)
	}

	return executeWarpToolsLoop(ctx, provider, req, cfg, onChunk)
}

func executeWarpToolsLoop(ctx context.Context, provider providers.Provider, req *providers.CompletionRequest, cfg *WarpToolsConfig, onChunk func(*providers.CompletionResponse)) (*providers.CompletionResponse, error) {
	currentReq := *req
	// Copy messages slice so we don't mutate caller's original slice
	currentReq.Messages = append([]providers.Message{}, req.Messages...)

	maxTurns := 4
	for turn := 0; turn < maxTurns; turn++ {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		// In intermediate search turns, run non-streaming to intercept searchTools calls
		resp, err := provider.Chat(ctx, &currentReq, nil)
		if err != nil {
			return nil, err
		}

		searchCalls := findToolCallsByName(resp.ToolCalls, "searchTools")
		if len(searchCalls) == 0 {
			// No searchTools called: unwrap any execute calls and return final response
			unwrappedResp := unwrapExecuteCalls(resp)
			if req.Stream && onChunk != nil {
				// Replay response to stream handler
				if unwrappedResp.Thought != "" {
					onChunk(&providers.CompletionResponse{Thought: unwrappedResp.Thought})
				}
				if unwrappedResp.Content != "" {
					onChunk(&providers.CompletionResponse{Content: unwrappedResp.Content})
				}
				if len(unwrappedResp.ToolCalls) > 0 {
					onChunk(&providers.CompletionResponse{ToolCalls: unwrappedResp.ToolCalls})
				}
			}
			return unwrappedResp, nil
		}

		// Model called searchTools: intercept and resolve locally using semantic search index
		log.Printf("[WARP_TOOLS] Turn %d: Intercepting %d searchTools call(s)", turn+1, len(searchCalls))

		// Append assistant response to messages
		currentReq.Messages = append(currentReq.Messages, providers.Message{
			Role:      "assistant",
			Content:   []providers.ContentPart{{Type: providers.ContentTypeText, Text: resp.Content}},
			ToolCalls: resp.ToolCalls,
		})

		var discoveredTools []providers.Tool
		existingToolNames := make(map[string]bool)
		for _, t := range currentReq.Tools {
			existingToolNames[t.Name] = true
		}

		for _, sc := range searchCalls {
			query := extractSearchQuery(sc.Function.Arguments)
			log.Printf("[WARP_TOOLS] Semantic query: %q", query)

			matches := cfg.Index.Search(query, cfg.TopK)
			log.Printf("[WARP_TOOLS] Found %d matching tool(s) for query %q", len(matches), query)

			var sb strings.Builder
			sb.WriteString(fmt.Sprintf("Search query: %q\nFound %d matching tools:\n\n", query, len(matches)))

			for i, m := range matches {
				sb.WriteString(fmt.Sprintf("--- Tool %d: %s ---\n", i+1, m.Name))
				sb.WriteString(CleanToolSchema(m))
				sb.WriteString("\n\n")

				if !existingToolNames[m.Name] {
					discoveredTools = append(discoveredTools, m)
					existingToolNames[m.Name] = true
				}
			}

			sb.WriteString("You may now call `execute(tools=[{\"name\": \"...\", \"arguments\": {...}}])` or invoke any discovered tool directly.")

			currentReq.Messages = append(currentReq.Messages, providers.Message{
				Role:       "tool",
				Name:       "searchTools",
				ToolCallID: sc.ID,
				Content:    []providers.ContentPart{{Type: providers.ContentTypeText, Text: sb.String()}},
			})
		}

		// Inject discovered tools into currentReq.Tools so the model can invoke them
		currentReq.Tools = append(currentReq.Tools, discoveredTools...)
	}

	// Final turn if maximum turns reached
	resp, err := provider.Chat(ctx, &currentReq, onChunk)
	if err != nil {
		return nil, err
	}
	return unwrapExecuteCalls(resp), nil
}

func findToolCallsByName(tcs []providers.ToolCall, name string) []providers.ToolCall {
	var matches []providers.ToolCall
	for _, tc := range tcs {
		if strings.EqualFold(tc.Function.Name, name) {
			matches = append(matches, tc)
		}
	}
	return matches
}

func extractSearchQuery(argsStr string) string {
	var argsObj map[string]any
	if err := json.Unmarshal([]byte(argsStr), &argsObj); err == nil {
		if q, ok := argsObj["query"].(string); ok && strings.TrimSpace(q) != "" {
			return strings.TrimSpace(q)
		}
		// Also check "q" or "search"
		if q, ok := argsObj["q"].(string); ok && strings.TrimSpace(q) != "" {
			return strings.TrimSpace(q)
		}
		if q, ok := argsObj["search"].(string); ok && strings.TrimSpace(q) != "" {
			return strings.TrimSpace(q)
		}
	}
	return strings.TrimSpace(argsStr)
}

type executeToolCallArg struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

func unwrapExecuteCalls(resp *providers.CompletionResponse) *providers.CompletionResponse {
	if resp == nil || len(resp.ToolCalls) == 0 {
		return resp
	}

	var unwrapped []providers.ToolCall
	for _, tc := range resp.ToolCalls {
		if strings.EqualFold(tc.Function.Name, "execute") {
			var parsed struct {
				Tools []executeToolCallArg `json:"tools"`
			}
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &parsed); err == nil && len(parsed.Tools) > 0 {
				for i, t := range parsed.Tools {
					argsBytes, _ := json.Marshal(t.Arguments)
					unwrapped = append(unwrapped, providers.ToolCall{
						ID:   fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), i),
						Type: "function",
						Function: providers.FunctionCall{
							Name:      t.Name,
							Arguments: string(argsBytes),
						},
					})
				}
				continue
			}

			// Also try single tool format {"name": "...", "arguments": {...}}
			var single executeToolCallArg
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &single); err == nil && single.Name != "" {
				argsBytes, _ := json.Marshal(single.Arguments)
				unwrapped = append(unwrapped, providers.ToolCall{
					ID:   tc.ID,
					Type: "function",
					Function: providers.FunctionCall{
						Name:      single.Name,
						Arguments: string(argsBytes),
					},
				})
				continue
			}
		}

		// Keep native non-execute tool calls
		unwrapped = append(unwrapped, tc)
	}

	resp.ToolCalls = unwrapped
	return resp
}
