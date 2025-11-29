package api

import (
	"fmt"
	"net/http"
)

type APIError struct {
	Code      int
	Message   string
	Err       error
	RequestID string
}

func (e *APIError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s: %v", e.Message, e.Err)
	}
	return e.Message
}

func NewError(code int, message string, err error) *APIError {
	return &APIError{
		Code:    code,
		Message: message,
		Err:     err,
	}
}

func NewErrorWithRequestID(code int, message string, err error, requestID string) *APIError {
	return &APIError{
		Code:      code,
		Message:   message,
		Err:       err,
		RequestID: requestID,
	}
}

var (
	ErrNotFound     = NewError(http.StatusNotFound, "resource not found", nil)
	ErrBadRequest   = NewError(http.StatusBadRequest, "bad request", nil)
	ErrUnauthorized = NewError(http.StatusUnauthorized, "unauthorized", nil)
	ErrInternal     = NewError(http.StatusInternalServerError, "internal server error", nil)
)

// WrapError wraps an error with request ID
func WrapError(err *APIError, requestID string) *APIError {
	if err == nil {
		return nil
	}
	return NewErrorWithRequestID(err.Code, err.Message, err.Err, requestID)
}
