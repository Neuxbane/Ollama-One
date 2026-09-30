# Build stage
FROM golang:1.26.2-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN go build -o ollama-one .

# Final stage
FROM alpine:latest

WORKDIR /app

COPY --from=builder /app/ollama-one .
COPY --from=builder /app/config.json .

EXPOSE 11434

CMD ["./ollama-one"]
