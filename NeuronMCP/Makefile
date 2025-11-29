.PHONY: build clean test run

build:
	go build -o bin/neurondb-mcp ./cmd/neurondb-mcp

clean:
	rm -rf bin/

test:
	go test ./...

run: build
	./bin/neurondb-mcp

