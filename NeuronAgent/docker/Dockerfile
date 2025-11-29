# Multi-stage build for NeuronAgent
FROM golang:1.23-bookworm AS builder

WORKDIR /build

# Copy go mod files first for better layer caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags="-w -s" -o agent-server ./cmd/agent-server

# Final stage - use Debian slim for better compatibility
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata \
    netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r neuronagent && useradd -r -g neuronagent -u 1000 neuronagent

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/agent-server .

# Copy migrations
COPY --from=builder /build/migrations ./migrations

# Copy configs directory if it exists
COPY --from=builder /build/configs ./configs

# Change ownership to non-root user
RUN chown -R neuronagent:neuronagent /app

# Switch to non-root user
USER neuronagent

EXPOSE 8080

# Add health check (using curl if available, otherwise basic TCP check)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD nc -z localhost 8080 || exit 1

CMD ["./agent-server"]

