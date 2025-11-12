\set ON_ERROR_STOP on
\timing on
\pset footer off

\if :{?sample_count}
\else
\set sample_count 1000000
\endif

\echo Using sample_count = :sample_count

DROP EXTENSION IF EXISTS neurondb CASCADE;
CREATE EXTENSION IF NOT EXISTS neurondb;

DROP TABLE IF EXISTS sample_data CASCADE;
DROP TABLE IF EXISTS sample_train CASCADE;
DROP TABLE IF EXISTS sample_test CASCADE;

DELETE FROM neurondb.ml_models
WHERE training_table IN ('sample_train', 'sample_test');

CREATE TABLE sample_data AS
SELECT
	gs AS sample_id,
	(
		SELECT array_agg((random() * 2 - 1)::float4 ORDER BY dim)
		FROM generate_series(1, 256) AS dim
	)::vector(256) AS features,
	(random() > 0.5)::int AS label
FROM generate_series(1, :sample_count) AS gs;

ANALYZE sample_data;

CREATE TABLE sample_train AS
SELECT *
FROM sample_data
WHERE sample_id % 5 <> 0;

CREATE TABLE sample_test AS
SELECT *
FROM sample_data
WHERE sample_id % 5 = 0;

ANALYZE sample_train;
ANALYZE sample_test;
