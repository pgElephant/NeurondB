package mcp

import (
	"bufio"
	"fmt"
	"io"
	"os"
)

// StdioTransport handles MCP communication over stdio
type StdioTransport struct {
	stdin  *bufio.Reader
	stdout io.Writer
	stderr io.Writer
}

// NewStdioTransport creates a new stdio transport
func NewStdioTransport() *StdioTransport {
	return &StdioTransport{
		stdin:  bufio.NewReader(os.Stdin),
		stdout: os.Stdout,
		stderr: os.Stderr,
	}
}

// ReadMessage reads a JSON-RPC message from stdin
func (t *StdioTransport) ReadMessage() (*JSONRPCRequest, error) {
	// Read Content-Length header
	var contentLength int
	for {
		line, err := t.stdin.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("failed to read header: %w", err)
		}

		if line == "\r\n" || line == "\n" {
			break
		}

		if _, err := fmt.Sscanf(line, "Content-Length: %d", &contentLength); err == nil {
			continue
		}
	}

	// Read message body
	body := make([]byte, contentLength)
	if _, err := io.ReadFull(t.stdin, body); err != nil {
		return nil, fmt.Errorf("failed to read message body: %w", err)
	}

	return ParseRequest(body)
}

// WriteMessage writes a JSON-RPC message to stdout
func (t *StdioTransport) WriteMessage(resp *JSONRPCResponse) error {
	data, err := SerializeResponse(resp)
	if err != nil {
		return fmt.Errorf("failed to serialize response: %w", err)
	}

	// Write Content-Length header
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(data))
	if _, err := t.stdout.Write([]byte(header)); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}

	// Write message body
	if _, err := t.stdout.Write(data); err != nil {
		return fmt.Errorf("failed to write body: %w", err)
	}

	return nil
}

// WriteError writes an error to stderr
func (t *StdioTransport) WriteError(err error) {
	fmt.Fprintf(t.stderr, "Error: %v\n", err)
}

