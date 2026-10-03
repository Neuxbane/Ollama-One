package main

import (
	"embed"
	"io/fs"
	"net/http"
)

// assetsFS embeds the static landing page and documentation assets directly into
// the compiled binary. This guarantees the UI is always available regardless of
// the process working directory (e.g. when running inside a minimal Docker image
// that only ships the binary).
//
//go:embed assets
var assetsFS embed.FS

// assetsSub returns a filesystem rooted at the embedded "assets" directory.
func assetsSub() fs.FS {
	sub, err := fs.Sub(assetsFS, "assets")
	if err != nil {
		// This should never happen because the embed directive guarantees the
		// directory exists at compile time.
		panic(err)
	}
	return sub
}

// staticHandler serves the embedded static assets (index.html, images, etc.).
func staticHandler() http.Handler {
	return http.FileServer(http.FS(assetsSub()))
}
