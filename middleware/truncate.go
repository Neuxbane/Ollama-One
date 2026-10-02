package middleware

import (
	"fmt"
	"log"
	"strconv"
	"strings"

	"github.com/Neuxbane/Ollama-One/providers"
)

func init() {
	Register("truncate", TruncateModifier)
}

// TruncateText truncates text to max `limit` characters (runes), appending a notice
// if truncation occurs. The total length of the returned string is guaranteed to be <= limit.
func TruncateText(s string, limit int) string {
	if limit <= 0 {
		return s
	}
	runes := []rune(s)
	if len(runes) <= limit {
		return s
	}

	overflow := len(runes) - limit
	marker := fmt.Sprintf("\n... [truncated %d chars]", overflow)
	markerRunes := []rune(marker)

	if limit > len(markerRunes) {
		keepCount := limit - len(markerRunes)
		return string(runes[:keepCount]) + marker
	}
	return string(runes[:limit])
}

// TruncateModifier implements the :truncate(target, limit) middleware.
// Target scopes:
// - "all": truncates system prompt, user messages, assistant messages, and tool/function responses & arguments
// - "tools", "tool", "function", "functions": truncates only tool/function responses and arguments
// - "user": truncates only user messages
// - "assistant": truncates only assistant messages
// - "system": truncates system messages and system instructions
//
// Default target: "all"
// Default limit: 4000
func TruncateModifier(req *providers.CompletionRequest, args []string) error {
	target := "all"
	limit := 4000

	if len(args) == 1 {
		if n, err := strconv.Atoi(args[0]); err == nil && n > 0 {
			limit = n
		} else if strings.TrimSpace(args[0]) != "" {
			target = strings.ToLower(strings.TrimSpace(args[0]))
		}
	} else if len(args) >= 2 {
		// Could be (target, limit) or (limit, target)
		if n, err := strconv.Atoi(args[1]); err == nil && n > 0 {
			target = strings.ToLower(strings.TrimSpace(args[0]))
			limit = n
		} else if n, err := strconv.Atoi(args[0]); err == nil && n > 0 {
			limit = n
			target = strings.ToLower(strings.TrimSpace(args[1]))
		} else {
			target = strings.ToLower(strings.TrimSpace(args[0]))
		}
	}

	truncateSystem := (target == "all" || target == "system")
	truncateTools := (target == "all" || target == "tools" || target == "tool" || target == "function" || target == "functions")
	truncateUser := (target == "all" || target == "user")
	truncateAssistant := (target == "all" || target == "assistant")

	// 1. Truncate SystemInstruction if applicable
	if truncateSystem && req.SystemInstruction != "" {
		origLen := len([]rune(req.SystemInstruction))
		if origLen > limit {
			req.SystemInstruction = TruncateText(req.SystemInstruction, limit)
			log.Printf("[MIDDLEWARE][TRUNCATE] Truncated SystemInstruction from %d to %d chars (max: %d)", origLen, len([]rune(req.SystemInstruction)), limit)
		}
	}

	// 2. Truncate Messages
	for i := range req.Messages {
		msg := &req.Messages[i]
		role := strings.ToLower(msg.Role)

		shouldTruncateContent := false
		switch role {
		case "tool", "function":
			shouldTruncateContent = truncateTools
		case "user":
			shouldTruncateContent = truncateUser
		case "assistant":
			shouldTruncateContent = truncateAssistant
		case "system", "developer":
			shouldTruncateContent = truncateSystem
		default:
			shouldTruncateContent = (target == "all")
		}

		// Truncate content parts (e.g. text or tool results)
		if shouldTruncateContent {
			for j := range msg.Content {
				part := &msg.Content[j]
				if part.Type == providers.ContentTypeText && part.Text != "" {
					origLen := len([]rune(part.Text))
					if origLen > limit {
						part.Text = TruncateText(part.Text, limit)
						desc := "input message"
						if role == "tool" || role == "function" {
							desc = fmt.Sprintf("function response (tool: %s, call_id: %s)", msg.Name, msg.ToolCallID)
						}
						log.Printf("[MIDDLEWARE][TRUNCATE] Truncated %s (role: %s) from %d to %d chars (max: %d)", desc, role, origLen, len([]rune(part.Text)), limit)
					}
				}
			}
		}

		// Truncate tool calls if tools scope matches
		if truncateTools && len(msg.ToolCalls) > 0 {
			for j := range msg.ToolCalls {
				tc := &msg.ToolCalls[j]
				if tc.Function.Arguments != "" {
					origLen := len([]rune(tc.Function.Arguments))
					if origLen > limit {
						tc.Function.Arguments = TruncateText(tc.Function.Arguments, limit)
						tc.Arguments = tc.Function.Arguments
						log.Printf("[MIDDLEWARE][TRUNCATE] Truncated tool call arguments (func: %s, id: %s) from %d to %d chars (max: %d)", tc.Function.Name, tc.ID, origLen, len([]rune(tc.Function.Arguments)), limit)
					}
				}
			}
		}
	}

	return nil
}
