package metrics

import (
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	// Request metrics
	httpRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_http_requests_total",
			Help: "Total number of HTTP requests",
		},
		[]string{"method", "endpoint", "status"},
	)

	httpRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "neurondb_agent_http_request_duration_seconds",
			Help:    "HTTP request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "endpoint"},
	)

	// Agent metrics
	agentExecutionsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_executions_total",
			Help: "Total number of agent executions",
		},
		[]string{"agent_id", "status"},
	)

	agentExecutionDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "neurondb_agent_execution_duration_seconds",
			Help:    "Agent execution duration in seconds",
			Buckets: []float64{0.1, 0.5, 1, 2, 5, 10, 30, 60},
		},
		[]string{"agent_id"},
	)

	// LLM metrics
	llmCallsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_llm_calls_total",
			Help: "Total number of LLM calls",
		},
		[]string{"model", "status"},
	)

	llmTokensTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_llm_tokens_total",
			Help: "Total number of LLM tokens",
		},
		[]string{"model", "type"},
	)

	// Memory metrics
	memoryChunksStored = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_memory_chunks_stored_total",
			Help: "Total number of memory chunks stored",
		},
		[]string{"agent_id"},
	)

	memoryRetrievalsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_memory_retrievals_total",
			Help: "Total number of memory retrievals",
		},
		[]string{"agent_id"},
	)

	// Tool metrics
	toolExecutionsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_tool_executions_total",
			Help: "Total number of tool executions",
		},
		[]string{"tool_name", "status"},
	)

	toolExecutionDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "neurondb_agent_tool_execution_duration_seconds",
			Help:    "Tool execution duration in seconds",
			Buckets: []float64{0.01, 0.1, 0.5, 1, 5, 10, 30},
		},
		[]string{"tool_name"},
	)

	// Job metrics
	jobsQueued = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "neurondb_agent_jobs_queued",
			Help: "Number of jobs in queue",
		},
	)

	jobsProcessedTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "neurondb_agent_jobs_processed_total",
			Help: "Total number of jobs processed",
		},
		[]string{"type", "status"},
	)
)

// RecordHTTPRequest records an HTTP request
func RecordHTTPRequest(method, endpoint string, status int, duration time.Duration) {
	httpRequestsTotal.WithLabelValues(method, endpoint, http.StatusText(status)).Inc()
	httpRequestDuration.WithLabelValues(method, endpoint).Observe(duration.Seconds())
}

// RecordAgentExecution records an agent execution
func RecordAgentExecution(agentID, status string, duration time.Duration) {
	agentExecutionsTotal.WithLabelValues(agentID, status).Inc()
	agentExecutionDuration.WithLabelValues(agentID).Observe(duration.Seconds())
}

// RecordLLMCall records an LLM call
func RecordLLMCall(model, status string, promptTokens, completionTokens int) {
	llmCallsTotal.WithLabelValues(model, status).Inc()
	llmTokensTotal.WithLabelValues(model, "prompt").Add(float64(promptTokens))
	llmTokensTotal.WithLabelValues(model, "completion").Add(float64(completionTokens))
}

// RecordMemoryChunkStored records a memory chunk being stored
func RecordMemoryChunkStored(agentID string) {
	memoryChunksStored.WithLabelValues(agentID).Inc()
}

// RecordMemoryRetrieval records a memory retrieval
func RecordMemoryRetrieval(agentID string) {
	memoryRetrievalsTotal.WithLabelValues(agentID).Inc()
}

// RecordToolExecution records a tool execution
func RecordToolExecution(toolName, status string, duration time.Duration) {
	toolExecutionsTotal.WithLabelValues(toolName, status).Inc()
	toolExecutionDuration.WithLabelValues(toolName).Observe(duration.Seconds())
}

// RecordJobQueued records a job being queued
func RecordJobQueued() {
	jobsQueued.Inc()
}

// RecordJobProcessed records a job being processed
func RecordJobProcessed(jobType, status string) {
	jobsProcessedTotal.WithLabelValues(jobType, status).Inc()
	jobsQueued.Dec()
}

// Handler returns the Prometheus metrics handler
func Handler() http.Handler {
	return promhttp.Handler()
}

