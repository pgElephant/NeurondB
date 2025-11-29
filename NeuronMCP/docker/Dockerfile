# Multi-stage build for NeuronMCP
FROM golang:1.23-bookworm AS builder

WORKDIR /build

# Copy go mod files first for better layer caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags="-w -s" -o neurondb-mcp ./cmd/neurondb-mcp

# Final stage - use Debian slim for better compatibility
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r neuronmcp && useradd -r -g neuronmcp -u 1000 neuronmcp

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/neurondb-mcp .

# Change ownership to non-root user
RUN chown -R neuronmcp:neuronmcp /app

# Switch to non-root user
USER neuronmcp

# MCP protocol uses stdio, so we don't expose ports
# Keep stdin/stdout/stderr available for MCP communication

# Set entrypoint to ensure stdio is properly configured
ENTRYPOINT ["./neurondb-mcp"]

