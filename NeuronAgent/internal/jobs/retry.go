package jobs

import (
	"context"
	"math"
	"time"
)

// RetryConfig configures retry behavior
type RetryConfig struct {
	MaxRetries      int
	InitialDelay    time.Duration
	MaxDelay        time.Duration
	BackoffMultiplier float64
	Jitter          bool
}

// DefaultRetryConfig returns default retry configuration
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries:       3,
		InitialDelay:     1 * time.Second,
		MaxDelay:         60 * time.Second,
		BackoffMultiplier: 2.0,
		Jitter:          true,
	}
}

// CalculateDelay calculates the delay for a retry attempt
func CalculateDelay(attempt int, config RetryConfig) time.Duration {
	// Exponential backoff: delay = initial * (multiplier ^ attempt)
	delay := float64(config.InitialDelay) * math.Pow(config.BackoffMultiplier, float64(attempt))
	
	// Cap at max delay
	if delay > float64(config.MaxDelay) {
		delay = float64(config.MaxDelay)
	}
	
	// Add jitter (random variation) to prevent thundering herd
	if config.Jitter {
		// Jitter: ±25% variation
		jitter := delay * 0.25
		delay = delay - jitter + (jitter * 2 * (float64(time.Now().UnixNano()%100) / 100))
	}
	
	return time.Duration(delay)
}

// ShouldRetry determines if a job should be retried based on error
func ShouldRetry(err error, attempt int, maxRetries int) bool {
	if attempt >= maxRetries {
		return false
	}
	
	// Don't retry on certain errors (e.g., validation errors)
	// In production, you'd check error types
	return true
}

// RetryWithBackoff retries a function with exponential backoff
func RetryWithBackoff(ctx context.Context, config RetryConfig, fn func() error) error {
	var lastErr error
	
	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		if err := fn(); err == nil {
			return nil
		} else {
			lastErr = err
		}
		
		// Don't wait after last attempt
		if attempt < config.MaxRetries {
			delay := CalculateDelay(attempt, config)
			
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
				// Continue to next attempt
			}
		}
	}
	
	return lastErr
}

