#!/bin/bash

# Build the Ollama-One binary
echo "Building Ollama-One..."
go build -o ollama-one .

if [ $? -eq 0 ]; then
    chmod +x ollama-one
    echo "Successfully built: ollama-one"
    ./ollama-one
else
    echo "Build failed!"
    exit 1
fi