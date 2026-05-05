package providers

import (
	"context"
	"sync/atomic"
)

// BaseProvider handles shared functionality like API key rotation
type BaseProvider struct {
	APIKeys         []string
	currentKeyIndex uint32
}

func (b *BaseProvider) GetNextKey() string {
	if len(b.APIKeys) == 0 {
		return ""
	}
	// Use atomic increment for thread-safe round-robin
	index := atomic.AddUint32(&b.currentKeyIndex, 1)
	return b.APIKeys[(index-1)%uint32(len(b.APIKeys))]
}

type ContentType string

const (
	ContentTypeText     ContentType = "text"
	ContentTypeImage    ContentType = "image"
	ContentTypeDocument ContentType = "document"
)

// ContentPart represents a single part of a message (text, image, or document)
type ContentPart struct {
	Type     ContentType `json:"type"`
	Text     string      `json:"text,omitempty"`
	Data     []byte      `json:"data,omitempty"`     // Base64 encoded or raw bytes
	FileURI  string      `json:"file_uri,omitempty"` // For large files (Gemini File API)
	MimeType string      `json:"mime_type,omitempty"`
}

// FunctionCall represents a function call within a tool call
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

// ToolCall represents a tool call made by the model
type ToolCall struct {
	ID       string        `json:"id,omitempty"`
	Type     string        `json:"type,omitempty"` // e.g., "function"
	Function FunctionCall  `json:"function,omitempty"`
	Name     string        `json:"name,omitempty"`
	Arguments string        `json:"arguments,omitempty"` // JSON string (for backwards compatibility)
}

// Tool represents a tool definition
type Tool struct {
	Type         string         `json:"type,omitempty"`
	Name         string         `json:"name,omitempty"`
	Description  string         `json:"description,omitempty"`
	Parameters   map[string]any `json:"parameters,omitempty"`
	Function     *FunctionCall  `json:"function,omitempty"` // For single function tool
	Functions    []Tool         `json:"functions,omitempty"` // For tools that contain multiple functions
	GoogleSearch bool           `json:"google_search,omitempty"` // Specific to Gemini
}

type ThinkingLevel string

const (
	ThinkingLevelLow    ThinkingLevel = "low"
	ThinkingLevelMedium ThinkingLevel = "medium"
	ThinkingLevelHigh   ThinkingLevel = "high"
)

type ThinkingConfig struct {
	IncludeThoughts bool          `json:"include_thoughts,omitempty"`
	ThinkingLevel   ThinkingLevel `json:"thinking_level,omitempty"`
	ThinkingBudget  int           `json:"thinking_budget,omitempty"` // Gemini-specific
}

// Message represents a single message in a chat conversation
type Message struct {
	Role      string        `json:"role"`
	Content   []ContentPart `json:"content"`
	ToolCalls []ToolCall    `json:"tool_calls,omitempty"`
}

// CompletionRequest defines the input for a chat completion
type CompletionRequest struct {
	Model             string          `json:"model"`
	SystemInstruction string          `json:"system_instruction,omitempty"`
	Messages          []Message       `json:"messages"`
	Tools             []Tool          `json:"tools,omitempty"`
	Stream            bool            `json:"stream"`
	Thinking          *ThinkingConfig `json:"thinking,omitempty"`
}

// CompletionResponse defines the output for a chat completion
type CompletionResponse struct {
	Content   string     `json:"content"`
	Thought   string     `json:"thought,omitempty"` // For thinking responses
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ModelInfo represents information about a specific model
type ModelInfo struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	ContextSize  int      `json:"context_size"`
	Capabilities []string `json:"capabilities"` // e.g., ["vision", "tools", "json"]
}

// Provider is the interface that all LLM providers must implement
type Provider interface {
	ListModels(ctx context.Context) ([]ModelInfo, error)
	Chat(ctx context.Context, req *CompletionRequest, onChunk func(*CompletionResponse)) (*CompletionResponse, error)
}
