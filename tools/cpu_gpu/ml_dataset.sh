#!/bin/bash

set -euo pipefail

REPO_URL="https://github.com/EpistasisLab/pmlb.git"
REPO_DIR="pmlb"

VERBOSE=0
DATASETS=""
DEFAULT_DATASET="adult"

usage() {
	echo "Usage: $0 [--dataset=<name>] [--verbose] [dataset ...]"
	echo "  --dataset=<name>     Specify a PMLB dataset (overrides positional)."
	echo "  --verbose            Enable verbose mode (default: off)."
	echo "  dataset ...          Datasets to import (space/comma separated)."
	echo "If no dataset is given, default is: $DEFAULT_DATASET"
	exit 1
}

parse_args() {
	while [ "$#" -gt 0 ]; do
		case "$1" in
			--help|-h)
				usage
				;;
			--verbose)
				VERBOSE=1
				;;
			--dataset=*)
				DATASETS="${1#*=}"
				;;
			-*)
				echo "Unknown option: $1" 1>&2
				usage
				;;
			*)
				if [ -z "$DATASETS" ]; then
					DATASETS="$1"
				else
					DATASETS="$DATASETS $1"
				fi
				;;
		esac
		shift
	done

	if [ -z "$DATASETS" ]; then
		DATASETS="$DEFAULT_DATASET"
	fi

	# Allow comma-separated datasets
	DATASETS=$(echo "$DATASETS" | tr ',' ' ')
}

log() {
	if [ "$VERBOSE" -eq 1 ]; then
		echo "$@"
	fi
}

# Clone dataset repository if necessary
clone_pmlb() {
	if [ ! -d "$REPO_DIR" ]; then
		log "Cloning PMLB dataset repository..."
		git clone "$REPO_URL" "$REPO_DIR"
	else
		log "PMLB repository already present."
	fi
}

# Find and validate a dataset by name
select_dataset() {
	DATASET="$1"
	CSV_FILE="${REPO_DIR}/datasets/${DATASET}/${DATASET}.tsv"
	if [ ! -f "$CSV_FILE" ]; then
		echo "Dataset file not found: $CSV_FILE"
		exit 1
	fi
	echo "$DATASET"
}

# Convert TSV to CSV, removing header
convert_tsv_to_csv() {
	local DATASET="$1"
	local CSV_FILE="$2"
	local CSV_OUT="/tmp/${DATASET}.csv"
	tail -n +2 "$CSV_FILE" | tr '\t' ',' > "$CSV_OUT"
	echo "$CSV_OUT"
}

# Ensure PostgreSQL is accepting connections
check_postgres() {
	local PGUSER="$1"
	local PGDATABASE="$2"
	if ! psql -U "$PGUSER" -d "$PGDATABASE" -c '\q' 2>/dev/null; then
		echo "PostgreSQL server is not accessible with user '$PGUSER' and db '$PGDATABASE'"
		exit 1
	fi
}

# Infer column definitions from the dataset header
infer_columns() {
	local CSV_FILE="$1"
	local HEADER
	HEADER=$(head -1 "$CSV_FILE")
	IFS=$'\t' read -r -a COLS <<< "$HEADER"
	local COL_DEF=""
	for col in "${COLS[@]}"; do
		if [[ "$col" == "class" ]]; then
			COL_DEF+="$col int,"
		else
			COL_DEF+="$col float4,"
		fi
	done
	COL_DEF=${COL_DEF%,}
	echo "$COL_DEF"
}

# Create or drop target table
create_table() {
	local PGUSER="$1"
	local PGDATABASE="$2"
	local TABLE="$3"
	local CREATE_COLS="$4"

	psql -U "$PGUSER" -d "$PGDATABASE" <<EOSQL
DROP TABLE IF EXISTS $TABLE CASCADE;
CREATE TABLE $TABLE (
	$CREATE_COLS
);
EOSQL
}

# Import data into PostgreSQL
import_csv() {
	local PGUSER="$1"
	local PGDATABASE="$2"
	local TABLE="$3"
	local CSV_OUT="$4"
	psql -U "$PGUSER" -d "$PGDATABASE" -c "\copy $TABLE FROM '$CSV_OUT' CSV"
}

######## MAIN ########

parse_args "$@"

PGUSER=${PGUSER:-postgres}
PGDATABASE=${PGDATABASE:-postgres}

clone_pmlb

for DATASET in $DATASETS; do
	DATASET=$(select_dataset "$DATASET")
	CSV_FILE="${REPO_DIR}/datasets/${DATASET}/${DATASET}.tsv"
	CSV_OUT=$(convert_tsv_to_csv "$DATASET" "$CSV_FILE")

	TABLE="pmlb_${DATASET}"

	check_postgres "$PGUSER" "$PGDATABASE"

	log "Dropping and creating table $TABLE..."
	CREATE_COLS=$(infer_columns "$CSV_FILE")
	create_table "$PGUSER" "$PGDATABASE" "$TABLE" "$CREATE_COLS"

	log "Importing CSV into $TABLE..."
	import_csv "$PGUSER" "$PGDATABASE" "$TABLE" "$CSV_OUT"

	echo "Table '$TABLE' successfully created and populated in PostgreSQL."
done
