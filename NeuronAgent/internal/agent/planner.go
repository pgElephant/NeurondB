package agent

import (
	"context"
	"fmt"
)

type Planner struct {
	maxIterations int
}

func NewPlanner() *Planner {
	return &Planner{
		maxIterations: 10, // Prevent infinite loops
	}
}

// Plan creates a multi-step plan for complex tasks
func (p *Planner) Plan(ctx context.Context, userMessage string, availableTools []string) ([]PlanStep, error) {
	// Simple implementation: single step plan
	// In production, this would use an LLM to break down complex tasks
	steps := []PlanStep{
		{
			Action:   "execute",
			Tool:    "",
			Payload: map[string]interface{}{"query": userMessage},
		},
	}
	return steps, nil
}

type PlanStep struct {
	Action  string
	Tool    string
	Payload map[string]interface{}
}

// ExecutePlan executes a multi-step plan
func (p *Planner) ExecutePlan(ctx context.Context, steps []PlanStep, executor func(step PlanStep) (interface{}, error)) ([]interface{}, error) {
	var results []interface{}
	iterations := 0

	for _, step := range steps {
		if iterations >= p.maxIterations {
			return results, fmt.Errorf("max iterations reached")
		}

		result, err := executor(step)
		if err != nil {
			return results, fmt.Errorf("step failed: %w", err)
		}

		results = append(results, result)
		iterations++
	}

	return results, nil
}

