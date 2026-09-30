# Build stage
FROM golang:1.26.2-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 go build -o ollama-one .

# Final stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /app

COPY --from=builder /app/ollama-one .
COPY --from=builder /app/config.json .

ENV HOST=0.0.0.0

EXPOSE 11434

CMD ["./ollama-one"]
