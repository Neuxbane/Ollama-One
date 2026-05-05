package main

import (
	"sync"
	"time"
	"github.com/Neuxbane/Ollama-One/providers"
)

type Session struct {
	ID        string
	Messages  []providers.Message
	LastSeen  time.Time
}

type SessionManager struct {
	sessions map[string]*Session
	mu       sync.RWMutex
}

func NewSessionManager() *SessionManager {
	return &SessionManager{
		sessions: make(map[string]*Session),
	}
}

func (sm *SessionManager) GetSession(id string) *Session {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if s, ok := sm.sessions[id]; ok {
		s.LastSeen = time.Now()
		return s
	}

	s := &Session{
		ID:       id,
		Messages: []providers.Message{},
		LastSeen: time.Now(),
	}
	sm.sessions[id] = s
	return s
}

func (sm *SessionManager) UpdateSession(id string, messages []providers.Message) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if s, ok := sm.sessions[id]; ok {
		s.Messages = messages
		s.LastSeen = time.Now()
	}
}
