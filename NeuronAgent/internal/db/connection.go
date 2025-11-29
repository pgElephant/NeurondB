package db

import (
	"context"
	"fmt"
	"time"

	"github.com/jmoiron/sqlx"
	_ "github.com/lib/pq"
)

type DB struct {
	*sqlx.DB
	poolConfig PoolConfig
}

type PoolConfig struct {
	MaxOpenConns    int
	MaxIdleConns    int
	ConnMaxLifetime time.Duration
	ConnMaxIdleTime time.Duration
}

func NewDB(connStr string, poolConfig PoolConfig) (*DB, error) {
	db, err := sqlx.Connect("postgres", connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to connect: %w", err)
	}

	db.SetMaxOpenConns(poolConfig.MaxOpenConns)
	db.SetMaxIdleConns(poolConfig.MaxIdleConns)
	db.SetConnMaxLifetime(poolConfig.ConnMaxLifetime)
	db.SetConnMaxIdleTime(poolConfig.ConnMaxIdleTime)

	return &DB{DB: db, poolConfig: poolConfig}, nil
}

func (d *DB) HealthCheck(ctx context.Context) error {
	var result int
	err := d.DB.GetContext(ctx, &result, "SELECT 1")
	return err
}

func (d *DB) Close() error {
	return d.DB.Close()
}
