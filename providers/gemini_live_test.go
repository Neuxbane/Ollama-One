package providers

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
)

func TestWriteWavHeader(t *testing.T) {
	pcmData := []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}
	sampleRate := 24000
	numChannels := 1
	bitsPerSample := 16

	wavBytes := WriteWavHeader(pcmData, sampleRate, numChannels, bitsPerSample)

	if len(wavBytes) != 44+len(pcmData) {
		t.Fatalf("expected length %d, got %d", 44+len(pcmData), len(wavBytes))
	}

	if string(wavBytes[0:4]) != "RIFF" {
		t.Errorf("expected RIFF, got %s", string(wavBytes[0:4]))
	}

	fileSizeMinus8 := binary.LittleEndian.Uint32(wavBytes[4:8])
	if fileSizeMinus8 != uint32(len(pcmData)+36) {
		t.Errorf("expected size %d, got %d", len(pcmData)+36, fileSizeMinus8)
	}

	if string(wavBytes[8:12]) != "WAVE" {
		t.Errorf("expected WAVE, got %s", string(wavBytes[8:12]))
	}

	if string(wavBytes[12:16]) != "fmt " {
		t.Errorf("expected 'fmt ', got %s", string(wavBytes[12:16]))
	}

	sr := binary.LittleEndian.Uint32(wavBytes[24:28])
	if sr != uint32(sampleRate) {
		t.Errorf("expected sample rate %d, got %d", sampleRate, sr)
	}

	channels := binary.LittleEndian.Uint16(wavBytes[22:24])
	if channels != uint16(numChannels) {
		t.Errorf("expected channels %d, got %d", numChannels, channels)
	}

	bits := binary.LittleEndian.Uint16(wavBytes[34:36])
	if bits != uint16(bitsPerSample) {
		t.Errorf("expected bits %d, got %d", bitsPerSample, bits)
	}

	if string(wavBytes[36:40]) != "data" {
		t.Errorf("expected 'data', got %s", string(wavBytes[36:40]))
	}

	dataSize := binary.LittleEndian.Uint32(wavBytes[40:44])
	if dataSize != uint32(len(pcmData)) {
		t.Errorf("expected data size %d, got %d", len(pcmData), dataSize)
	}

	if !bytes.Equal(wavBytes[44:], pcmData) {
		t.Errorf("payload mismatch")
	}
}

func TestListModels(t *testing.T) {
	provider := NewGeminiLiveProvider([]string{"dummy-key"})
	models, err := provider.ListModels(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(models) == 0 {
		t.Fatalf("expected models, got 0")
	}

	foundPreview := false
	for _, m := range models {
		if m.ID == "gemini-3.1-flash-live-preview" {
			foundPreview = true
			break
		}
	}
	if !foundPreview {
		t.Errorf("expected gemini-3.1-flash-live-preview in models")
	}
}

func TestFormatHistory(t *testing.T) {
	messages := []Message{
		{
			Role: "user",
			Content: []ContentPart{
				{Type: ContentTypeText, Text: "Hello, who are you?"},
			},
		},
		{
			Role: "assistant",
			Content: []ContentPart{
				{Type: ContentTypeText, Text: "I am Gemini Live."},
			},
			ToolCalls: []ToolCall{
				{
					ID:   "call_1",
					Type: "function",
					Function: FunctionCall{
						Name:      "get_time",
						Arguments: `{"timezone":"UTC"}`,
					},
				},
			},
		},
	}

	formatted := FormatHistory(messages)
	if !strings.Contains(formatted, "CONVERSATION HISTORY") {
		t.Errorf("expected history header, got %s", formatted)
	}
	if !strings.Contains(formatted, "User: Hello, who are you?") {
		t.Errorf("expected user message, got %s", formatted)
	}
	if !strings.Contains(formatted, "Assistant: I am Gemini Live.") {
		t.Errorf("expected assistant message, got %s", formatted)
	}
	if !strings.Contains(formatted, "Called Tool: get_time with arguments: {\"timezone\":\"UTC\"}") {
		t.Errorf("expected tool call info, got %s", formatted)
	}
}

func TestGeminiLiveProviderChatMockWS(t *testing.T) {
	upgrader := websocket.Upgrader{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Logf("upgrade error: %v", err)
			return
		}
		defer conn.Close()

		// 1. Read setup frame
		_, setupBytes, err := conn.ReadMessage()
		if err != nil {
			t.Logf("read setup error: %v", err)
			return
		}
		var setupMsg liveSetupMessage
		if err := json.Unmarshal(setupBytes, &setupMsg); err != nil {
			t.Logf("unmarshal setup error: %v", err)
			return
		}

		// 2. Respond with setupComplete
		conn.WriteJSON(map[string]any{
			"setupComplete": map[string]any{},
		})

		// 3. Read clientContent
		_, clientContentBytes, err := conn.ReadMessage()
		if err != nil {
			t.Logf("read clientContent error: %v", err)
			return
		}
		var clientContentMsg liveClientContentMessage
		if err := json.Unmarshal(clientContentBytes, &clientContentMsg); err != nil {
			t.Logf("unmarshal clientContent error: %v", err)
			return
		}

		// 4. Send thought chunk
		conn.WriteJSON(map[string]any{
			"serverContent": map[string]any{
				"modelTurn": map[string]any{
					"parts": []map[string]any{
						{
							"thought": true,
							"text":    "Thinking about greetings...",
						},
					},
				},
			},
		})

		// 5. Send outputTranscription (audio transcription text)
		conn.WriteJSON(map[string]any{
			"serverContent": map[string]any{
				"outputTranscription": map[string]any{
					"text": "Hello from Gemini Live!",
				},
			},
		})

		// 6. Send turnComplete
		conn.WriteJSON(map[string]any{
			"serverContent": map[string]any{
				"turnComplete": true,
			},
		})
	}))
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http")

	provider := NewGeminiLiveProvider([]string{"dummy-key"})
	provider.WSURL = wsURL
	provider.OutputDir = t.TempDir()

	req := &CompletionRequest{
		Model: "gemini-3.1-flash-live-preview",
		Messages: []Message{
			{
				Role: "user",
				Content: []ContentPart{
					{Type: ContentTypeText, Text: "Say hello!"},
				},
			},
		},
		Stream: true,
	}

	var chunks []string
	var thoughts []string
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := provider.Chat(ctx, req, func(c *CompletionResponse) {
		if c.Content != "" {
			chunks = append(chunks, c.Content)
		}
		if c.Thought != "" {
			thoughts = append(thoughts, c.Thought)
		}
	})

	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	if resp.Content != "Hello from Gemini Live!" {
		t.Errorf("expected content 'Hello from Gemini Live!', got %q", resp.Content)
	}
	if resp.Thought != "Thinking about greetings..." {
		t.Errorf("expected thought 'Thinking about greetings...', got %q", resp.Thought)
	}
	if len(chunks) == 0 || chunks[0] != "Hello from Gemini Live!" {
		t.Errorf("expected chunk 'Hello from Gemini Live!', got %v", chunks)
	}
	if len(thoughts) == 0 || thoughts[0] != "Thinking about greetings..." {
		t.Errorf("expected thought chunk 'Thinking about greetings...', got %v", thoughts)
	}
}
