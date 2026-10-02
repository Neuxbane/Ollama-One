package middleware

import (
	"fmt"
	"log"
	"regexp"
	"strings"
	"sync"

	"github.com/Neuxbane/Ollama-One/providers"
)

// ModifierFunc represents a model middleware/function that transforms a completion request.
type ModifierFunc func(req *providers.CompletionRequest, args []string) error

// Directive represents a parsed model directive like `:name(arg1, arg2)`.
type Directive struct {
	Name string
	Args []string
	Raw  string
}

var (
	registryMu sync.RWMutex
	registry   = make(map[string]ModifierFunc)

	// directiveRegex matches patterns like :function(arg1, arg2)
	directiveRegex = regexp.MustCompile(`:([a-zA-Z0-9_\-]+)\s*\(([^)]*)\)`)
)

func init() {
	// Hook into providers.PreChatHook so middleware runs automatically on any Chat call
	providers.PreChatHook = Process
}

// Register registers a new model modifier by name.
func Register(name string, fn ModifierFunc) {
	registryMu.Lock()
	defer registryMu.Unlock()
	registry[strings.ToLower(name)] = fn
}

// Get returns the modifier function registered for the given name.
func Get(name string) (ModifierFunc, bool) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	fn, ok := registry[strings.ToLower(name)]
	return fn, ok
}

// ParseModelDirectives extracts all :function(args) directives from a model identifier
// and returns the clean base model name along with the parsed directives.
func ParseModelDirectives(rawModel string) (string, []Directive) {
	var directives []Directive

	matches := directiveRegex.FindAllStringSubmatch(rawModel, -1)
	for _, m := range matches {
		if len(m) >= 3 {
			name := strings.TrimSpace(m[1])
			rawArgs := strings.TrimSpace(m[2])
			args := parseDirectiveArgs(rawArgs)
			directives = append(directives, Directive{
				Name: name,
				Args: args,
				Raw:  m[0],
			})
		}
	}

	cleaned := directiveRegex.ReplaceAllString(rawModel, "")
	cleaned = strings.TrimSpace(cleaned)

	return cleaned, directives
}

func parseDirectiveArgs(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	var args []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		p = strings.Trim(p, `"'`)
		p = strings.TrimSpace(p)
		if p != "" {
			args = append(args, p)
		}
	}
	return args
}

// Process inspects req.Model for any :function(args) directives,
// executes each registered modifier in order, and updates req.Model to the clean base model.
func Process(req *providers.CompletionRequest) error {
	if req == nil || req.Model == "" {
		return nil
	}

	baseModel, directives := ParseModelDirectives(req.Model)
	if len(directives) == 0 {
		return nil
	}

	req.Model = baseModel

	for _, d := range directives {
		fn, ok := Get(d.Name)
		if !ok {
			log.Printf("[WARN][MIDDLEWARE] Unknown model directive %q (args: %v)", d.Name, d.Args)
			continue
		}
		log.Printf("[INFO][MIDDLEWARE] Applying :%s(%s) to model request (base model: %s)", d.Name, strings.Join(d.Args, ", "), baseModel)
		if err := fn(req, d.Args); err != nil {
			return fmt.Errorf("middleware %s failed: %w", d.Name, err)
		}
	}

	return nil
}
