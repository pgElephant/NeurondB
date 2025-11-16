#!/usr/bin/env python3

"""
higgs_uci_to_sql.py:
Download the HIGGS dataset from UCI and generate:

  - schema.sql
  - data.sql

Load into PostgreSQL with:
  psql -d yourdb -f higgs_sql/schema.sql
  psql -d yourdb -f higgs_sql/data.sql

UCI HIGGS format:
  - File: HIGGS.csv
  - ~11M rows
  - 29 columns
    col0   = label (0 or 1)
    col1-28 = real-valued features
"""

import argparse
import csv
import logging
import os
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from typing import List, Tuple
from urllib.request import urlopen
import time
from typing import Optional


UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/280/higgs.zip"
HIGGS_CSV_BASENAME = "HIGGS.csv"
EXPECTED_NUM_COLUMNS = 29
DEFAULT_LOG_EVERY_ROWS = 1_000_000
MB = 1024 * 1024


def setup_logging(verbose: bool) -> None:
    """
    Configure root logger based on verbosity.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclass
class HiggsConfig:
    """
    Container for CLI configuration.
    """
    output_dir: str
    table_name: str
    limit_rows: int
    reuse_csv: str
    verbose: bool
    log_every_rows: int = DEFAULT_LOG_EVERY_ROWS


def validate_table_name(table_name: str) -> str:
    """
    Validate a PostgreSQL unquoted identifier (simple heuristic).
    """
    if not table_name:
        raise ValueError("table name must be non-empty")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
        raise ValueError(
            "table name must match ^[A-Za-z_][A-Za-z0-9_]*$ (unquoted identifier)"
        )
    return table_name


def download_file(url: str, dest_path: str) -> None:
    """
    Download a file from URL to dest_path using streaming.
    """
    logging.info("Downloading %s", url)
    with urlopen(url) as resp, open(dest_path, "wb") as out:
        chunk_size = 1 * MB
        total = 0
        total_size = 0
        try:
            hdr_len = resp.getheader("Content-Length")
            total_size = int(hdr_len) if hdr_len else 0
        except Exception:
            total_size = 0

        start_ts = last_ts = time.time()
        last_reported_percent = -1
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
            now = time.time()

            # Progress calculation (if content-length available)
            if total_size > 0:
                percent = int((total * 100) / total_size)
            else:
                percent = -1

            # Throttle progress updates to ~2Hz or on new percent step
            if percent != last_reported_percent or (now - last_ts) >= 0.5:
                elapsed = max(0.001, now - start_ts)
                speed = total / elapsed  # bytes/sec
                mb_done = total / MB
                if total_size > 0:
                    mb_total = total_size / MB
                    sys.stderr.write(
                        f"\r[DOWNLOAD] {percent:3d}% "
                        f"({mb_done:.1f}/{mb_total:.1f} MB) "
                        f"@ {speed/MB:.2f} MB/s"
                    )
                else:
                    sys.stderr.write(
                        f"\r[DOWNLOAD] {mb_done:.1f} MB "
                        f"@ {speed/MB:.2f} MB/s"
                    )
                sys.stderr.flush()
                last_ts = now
                last_reported_percent = percent
        # Finish line
        sys.stderr.write("\n")
        sys.stderr.flush()
    logging.info("Download complete: %s", dest_path)


def extract_higgs_csv(zip_path: str, output_dir: str) -> str:
    """
    Extract HIGGS.csv from the zip file into output_dir.
    Return the path to HIGGS.csv.
    """
    logging.info("Extracting from %s", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        csv_name = None
        for n in names:
            lower = n.lower()
            if lower.endswith(".csv") and "higgs" in lower:
                csv_name = n
                break

        if csv_name is None:
            raise FileNotFoundError("No HIGGS CSV file found in zip.")

        target_path = os.path.join(output_dir, HIGGS_CSV_BASENAME)
        with zf.open(csv_name, "r") as src, open(target_path, "wb") as dst:
            while True:
                chunk = src.read(1 * MB)
                if not chunk:
                    break
                dst.write(chunk)

    logging.info("Extracted CSV to %s", target_path)
    return target_path


def find_local_higgs_csv(output_dir: str) -> Optional[str]:
    """
    Try to locate an existing HIGGS.csv in common locations.
    """
    candidates = [
        os.path.join(output_dir, HIGGS_CSV_BASENAME),
        os.path.join("tests", "datasets", HIGGS_CSV_BASENAME),
        os.path.join("datasets", HIGGS_CSV_BASENAME),
        HIGGS_CSV_BASENAME,
    ]
    for path in candidates:
        if os.path.isfile(path):
            logging.info("Found local CSV: %s", path)
            return path
    return None


def find_local_higgs_zip() -> Optional[str]:
    """
    Try to locate a local higgs.zip in common locations.
    """
    candidates = [
        os.path.join("tests", "datasets", "higgs.zip"),
        os.path.join("datasets", "higgs.zip"),
        "higgs.zip",
    ]
    for path in candidates:
        if os.path.isfile(path):
            logging.info("Found local ZIP: %s", path)
            return path
    return None


def build_columns() -> List[Tuple[str, str]]:
    """
    Define columns for HIGGS: label int, f1..f28 double precision.
    """
    cols: List[Tuple[str, str]] = [("label", "integer")]
    for i in range(1, 29):
        cols.append((f"f{i}", "double precision"))
    return cols


def write_schema_sql(path: str, table_name: str,
                     columns: List[Tuple[str, str]]) -> None:
    """
    Write schema.sql for the target table.
    """
    logging.info("Writing schema to %s", path)
    with open(path, "w", encoding="utf8") as f:
        f.write(f"DROP TABLE IF EXISTS {table_name};\n")
        f.write(f"CREATE TABLE {table_name} (\n")
        for idx, (name, pg_type) in enumerate(columns):
            comma = "," if idx < len(columns) - 1 else ""
            f.write(f'    "{name}" {pg_type}{comma}\n')
        f.write(");\n")
    logging.info("schema.sql written")


def escape_copy_value(value: str) -> str:
    """
    Escape value for COPY text format.
    """
    if value == "" or value is None:
        return r"\N"

    s = str(value)
    s = s.replace("\\", "\\\\")
    s = s.replace("\t", "\\t")
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    return s


def write_data_sql(
    path: str,
    table_name: str,
    columns: List[Tuple[str, str]],
    csv_path: str,
    limit_rows: int = 0,
    log_every_rows: int = DEFAULT_LOG_EVERY_ROWS,
) -> int:
    """
    Read HIGGS.csv and write data.sql using COPY FROM stdin.
    Returns number of rows written.
    """
    logging.info("Writing data to %s", path)
    col_names = [name for name, _ in columns]
    col_list = ", ".join(f'"{name}"' for name in col_names)

    total_written = 0

    with open(path, "w", encoding="utf8") as out_f:
        out_f.write(f"COPY {table_name} ({col_list}) FROM stdin;\n")

        with open(csv_path, "r", encoding="utf8") as csv_f:
            reader = csv.reader(csv_f)
            for row_idx, row in enumerate(reader):
                if limit_rows and total_written >= limit_rows:
                    break

                if len(row) != EXPECTED_NUM_COLUMNS:
                    logging.warning(
                        "Row %d has %d cols, expected %d. Skipping.",
                        row_idx, len(row), EXPECTED_NUM_COLUMNS
                    )
                    continue

                label = row[0].strip()
                feats = [x.strip() for x in row[1:]]

                values = [label] + feats
                esc = [escape_copy_value(v) for v in values]
                line = "\t".join(esc)
                out_f.write(line + "\n")
                total_written += 1

                if log_every_rows and total_written % log_every_rows == 0:
                    logging.info("Written %d rows", total_written)

        out_f.write("\\.\n")

    logging.info("data.sql written, rows: %d", total_written)
    return total_written


def ensure_dir(path: str) -> None:
    """
    Create directory if it does not exist.
    """
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        logging.info("Created directory: %s", path)


def build_paths(out_dir: str) -> Tuple[str, str]:
    """
    Compute output file paths.
    """
    schema_path = os.path.join(out_dir, "schema.sql")
    data_path = os.path.join(out_dir, "data.sql")
    return schema_path, data_path


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate schema.sql and data.sql from UCI HIGGS."
    )
    parser.add_argument(
        "--output-dir",
        default="higgs_sql",
        help="Directory for schema.sql and data.sql.",
    )
    parser.add_argument(
        "--table-name",
        default="higgs",
        help="Target PostgreSQL table name (unquoted identifier).",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="Limit number of rows in data.sql. 0 means full file.",
    )
    parser.add_argument(
        "--reuse-csv",
        default="",
        help="Existing HIGGS.csv path. If set, skip download.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY_ROWS,
        help="Log progress every N rows (default: 1,000,000).",
    )
    return parser.parse_args()


def generate_sql(cfg: HiggsConfig) -> Tuple[str, str]:
    """
    End-to-end generation of schema.sql and data.sql.
    Returns (schema_path, data_path).
    """
    ensure_dir(cfg.output_dir)
    schema_path, data_path = build_paths(cfg.output_dir)

    # Acquire CSV: prefer existing local artifacts (CSV or ZIP), then download.
    if cfg.reuse_csv:
        csv_path = cfg.reuse_csv
        logging.info("Reusing existing CSV: %s", csv_path)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError("--reuse-csv path does not exist")
    else:
        csv_local = find_local_higgs_csv(cfg.output_dir)
        if csv_local:
            csv_path = csv_local
        else:
            zip_local = find_local_higgs_zip()
            if zip_local:
                csv_path_extracted = extract_higgs_csv(zip_local, cfg.output_dir)
                csv_path = csv_path_extracted
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, "higgs.zip")
                    download_file(UCI_ZIP_URL, zip_path)
                    csv_path_extracted = extract_higgs_csv(zip_path, tmpdir)

                    final_csv = os.path.join(cfg.output_dir, HIGGS_CSV_BASENAME)
                    os.replace(csv_path_extracted, final_csv)
                    csv_path = final_csv
                    logging.info("CSV stored at %s", csv_path)

    # Columns and schema
    columns = build_columns()
    logging.debug("Columns: %s", columns)

    write_schema_sql(schema_path, cfg.table_name, columns)
    total_rows = write_data_sql(
        data_path, cfg.table_name, columns, csv_path,
        limit_rows=cfg.limit_rows,
        log_every_rows=cfg.log_every_rows,
    )

    logging.info("All files written. Total rows: %d", total_rows)
    logging.info("Load with:")
    logging.info("  psql -d yourdb -f %s", schema_path)
    logging.info("  psql -d yourdb -f %s", data_path)

    return schema_path, data_path


def main() -> int:
    args = parse_args()
    try:
        setup_logging(args.verbose)
        cfg = HiggsConfig(
            output_dir=args.output_dir,
            table_name=validate_table_name(args.table_name),
            limit_rows=max(0, int(args.limit_rows)),
            reuse_csv=str(args.reuse_csv).strip(),
            verbose=bool(args.verbose),
            log_every_rows=max(0, int(args.log_every)),
        )
        generate_sql(cfg)
        return 0
    except Exception as exc:
        logging.error("Failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

