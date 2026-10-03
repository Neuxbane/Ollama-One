package middleware

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/Neuxbane/Ollama-One/providers"
)

const defaultDenseDimension = 128

// Common API / function synonyms mapped to semantic canonical tokens
var semanticSynonymMap = map[string]string{
	"lookup":     "find",
	"search":     "find",
	"query":      "find",
	"retrieve":   "get",
	"fetch":      "get",
	"read":       "get",
	"select":     "get",
	"load":       "get",
	"inspect":    "get",
	"add":        "create",
	"insert":     "create",
	"new":        "create",
	"post":       "create",
	"make":       "create",
	"generate":   "create",
	"build":      "create",
	"delete":     "remove",
	"drop":       "remove",
	"destroy":    "remove",
	"clear":      "remove",
	"update":     "modify",
	"patch":      "modify",
	"edit":       "modify",
	"change":     "modify",
	"alter":      "modify",
	"set":        "modify",
	"exec":       "run",
	"start":      "run",
	"execute":    "run",
	"trigger":    "run",
	"call":       "run",
	"database":   "db",
	"sql":        "db",
	"table":      "db",
	"document":   "doc",
	"file":       "doc",
	"mail":       "email",
	"message":    "msg",
	"notify":     "alert",
	"notification": "alert",
	"weather":    "forecast",
	"temperature": "forecast",
	"climate":    "forecast",
	"calc":       "math",
	"calculate":  "math",
	"compute":    "math",
}

// ToolDoc holds the indexed metadata and vector for a tool
type ToolDoc struct {
	Tool       providers.Tool
	Tokens     []string
	TokenSet   map[string]bool
	Vector     []float64
	RawText    string
}

// SemanticToolIndex provides fast local semantic search across tool definitions
type SemanticToolIndex struct {
	Docs      []ToolDoc
	Dimension int
}

// NewSemanticToolIndex builds an in-memory semantic vector index from tools
func NewSemanticToolIndex(tools []providers.Tool) *SemanticToolIndex {
	idx := &SemanticToolIndex{
		Dimension: defaultDenseDimension,
	}

	for _, tool := range tools {
		doc := idx.buildDoc(tool)
		idx.Docs = append(idx.Docs, doc)
	}

	return idx
}

func tokenizeText(text string) []string {
	// Split camelCase and snake_case
	var words []string
	var current strings.Builder

	for i, r := range text {
		if r == '_' || r == '-' || r == '.' || r == '/' || r == ':' {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
			continue
		}

		if unicode.IsUpper(r) {
			if current.Len() > 0 && i > 0 && unicode.IsLower(rune(text[i-1])) {
				words = append(words, current.String())
				current.Reset()
			}
		}

		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			current.WriteRune(unicode.ToLower(r))
		} else {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
		}
	}
	if current.Len() > 0 {
		words = append(words, current.String())
	}

	var normalized []string
	for _, w := range words {
		w = strings.TrimSpace(w)
		if len(w) <= 1 {
			continue
		}
		// Apply synonym canonicalization
		if canon, ok := semanticSynonymMap[w]; ok {
			normalized = append(normalized, canon)
		}
		normalized = append(normalized, w)
	}
	return normalized
}

func hashString(s string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32()
}

func embedTokens(tokens []string, dim int, weight float64) []float64 {
	vec := make([]float64, dim)
	for _, tok := range tokens {
		// Embed whole token
		h := hashString(tok)
		idx := int(h % uint32(dim))
		sign := 1.0
		if (h>>16)%2 == 1 {
			sign = -1.0
		}
		vec[idx] += sign * weight

		// Embed character 3-grams for morphological/subword matching
		runes := []rune(tok)
		if len(runes) >= 3 {
			for i := 0; i <= len(runes)-3; i++ {
				ngram := string(runes[i : i+3])
				nh := hashString(ngram)
				nidx := int(nh % uint32(dim))
				nsign := 1.0
				if (nh>>16)%2 == 1 {
					nsign = -1.0
				}
				vec[nidx] += nsign * (weight * 0.5)
			}
		}
	}
	return vec
}

func normalizeVector(vec []float64) {
	var sumSq float64
	for _, v := range vec {
		sumSq += v * v
	}
	if sumSq == 0 {
		return
	}
	norm := math.Sqrt(sumSq)
	for i := range vec {
		vec[i] /= norm
	}
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot float64
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

func (idx *SemanticToolIndex) buildDoc(tool providers.Tool) ToolDoc {
	nameTokens := tokenizeText(tool.Name)
	descTokens := tokenizeText(tool.Description)

	var paramTokens []string
	if tool.Parameters != nil {
		if props, ok := tool.Parameters["properties"].(map[string]any); ok {
			for propName, propVal := range props {
				paramTokens = append(paramTokens, tokenizeText(propName)...)
				if propMap, ok := propVal.(map[string]any); ok {
					if pDesc, ok := propMap["description"].(string); ok {
						paramTokens = append(paramTokens, tokenizeText(pDesc)...)
					}
				}
			}
		}
	}

	allTokens := append([]string{}, nameTokens...)
	allTokens = append(allTokens, descTokens...)
	allTokens = append(allTokens, paramTokens...)

	tokenSet := make(map[string]bool)
	for _, t := range allTokens {
		tokenSet[t] = true
	}

	// Weighted vector: Tool Name (3x), Description (2x), Parameters (1x)
	vec := make([]float64, idx.Dimension)
	nameVec := embedTokens(nameTokens, idx.Dimension, 3.0)
	descVec := embedTokens(descTokens, idx.Dimension, 2.0)
	paramVec := embedTokens(paramTokens, idx.Dimension, 1.0)

	for i := 0; i < idx.Dimension; i++ {
		vec[i] = nameVec[i] + descVec[i] + paramVec[i]
	}
	normalizeVector(vec)

	raw := fmt.Sprintf("%s: %s", tool.Name, tool.Description)
	return ToolDoc{
		Tool:     tool,
		Tokens:   allTokens,
		TokenSet: tokenSet,
		Vector:   vec,
		RawText:  raw,
	}
}

// Search ranks indexed tools against the query and returns the top K matching tools
func (idx *SemanticToolIndex) Search(query string, topK int) []providers.Tool {
	if len(idx.Docs) == 0 {
		return nil
	}
	if topK <= 0 {
		topK = 5
	}
	if topK > len(idx.Docs) {
		topK = len(idx.Docs)
	}

	queryTokens := tokenizeText(query)
	if len(queryTokens) == 0 {
		// Return first topK tools if query is blank
		var fallback []providers.Tool
		for i := 0; i < topK; i++ {
			fallback = append(fallback, idx.Docs[i].Tool)
		}
		return fallback
	}

	// Compute dense query vector
	queryVec := embedTokens(queryTokens, idx.Dimension, 1.0)
	normalizeVector(queryVec)

	// Try local Ollama/OpenAI embedding endpoint if configured
	localEmbedVec := tryLocalEmbedding(query)
	if len(localEmbedVec) > 0 {
		// Optional external embedding boost if available
	}

	type searchHit struct {
		Doc   *ToolDoc
		Score float64
	}

	var hits []searchHit
	for i := range idx.Docs {
		doc := &idx.Docs[i]

		// 1. Cosine similarity of dense embeddings
		cosSim := cosineSimilarity(queryVec, doc.Vector)

		// 2. Exact keyword / token overlap
		matches := 0
		for _, qTok := range queryTokens {
			if doc.TokenSet[qTok] {
				matches++
			}
		}
		keywordScore := 0.0
		if len(queryTokens) > 0 {
			keywordScore = float64(matches) / float64(len(queryTokens))
		}

		// Combined hybrid score
		score := (0.6 * cosSim) + (0.4 * keywordScore)

		// Boost if query matches tool name substring
		lowerQuery := strings.ToLower(query)
		lowerName := strings.ToLower(doc.Tool.Name)
		if strings.Contains(lowerQuery, lowerName) || strings.Contains(lowerName, lowerQuery) {
			score += 0.3
		}

		hits = append(hits, searchHit{Doc: doc, Score: score})
	}

	sort.Slice(hits, func(i, j int) bool {
		return hits[i].Score > hits[j].Score
	})

	var result []providers.Tool
	for i := 0; i < topK && i < len(hits); i++ {
		result = append(result, hits[i].Doc.Tool)
	}

	return result
}

// tryLocalEmbedding checks if an Ollama or local embedding endpoint is running
func tryLocalEmbedding(text string) []float64 {
	embedURL := os.Getenv("OLLAMA_EMBED_URL")
	if embedURL == "" {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	body, _ := json.Marshal(map[string]any{
		"model":  "nomic-embed-text",
		"prompt": text,
	})

	req, err := http.NewRequestWithContext(ctx, "POST", embedURL, bytes.NewReader(body))
	if err != nil {
		return nil
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 300 * time.Millisecond}
	resp, err := client.Do(req)
	if err != nil || resp.StatusCode != http.StatusOK {
		if resp != nil {
			resp.Body.Close()
		}
		return nil
	}
	defer resp.Body.Close()

	var data struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err == nil {
		return data.Embedding
	}
	return nil
}

// CleanToolSchema generates a clean, concise JSON schema description for discovered tools
func CleanToolSchema(t providers.Tool) string {
	b, err := json.MarshalIndent(map[string]any{
		"name":        t.Name,
		"description": t.Description,
		"parameters":  t.Parameters,
	}, "", "  ")
	if err != nil {
		return fmt.Sprintf("%s: %s", t.Name, t.Description)
	}
	return string(b)
}
