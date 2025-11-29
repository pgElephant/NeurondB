package auth

import (
	"golang.org/x/crypto/bcrypt"
)

const bcryptCost = 12

// HashAPIKey hashes an API key using bcrypt
func HashAPIKey(key string) (string, error) {
	hash, err := bcrypt.GenerateFromPassword([]byte(key), bcryptCost)
	if err != nil {
		return "", err
	}
	return string(hash), nil
}

// VerifyAPIKey verifies an API key against its hash
func VerifyAPIKey(key, hash string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(key))
	return err == nil
}

// GetKeyPrefix returns the first 8 characters of a key for identification
func GetKeyPrefix(key string) string {
	if len(key) < 8 {
		return key
	}
	return key[:8]
}

