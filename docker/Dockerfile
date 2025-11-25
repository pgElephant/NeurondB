# syntax=docker/dockerfile:1.6
#
# NeuronDB reference image (CPU)
# Multi-stage build that compiles the extension against PostgreSQL and bundles
# ONNX Runtime for HuggingFace integration.
#
# Supports:
#   - PostgreSQL: 16, 17, 18 (via PG_MAJOR build arg)
#   - Architectures: amd64, arm64
#   - OS: Debian Bookworm (via postgres base image)
#
# Build args:
#   PG_MAJOR: PostgreSQL major version (16, 17, or 18, default: 17)
#   ONNX_VERSION: ONNX Runtime version (default: 1.17.0)

ARG PG_MAJOR=17
ARG ONNX_VERSION=1.17.0

FROM postgres:${PG_MAJOR}-bookworm AS builder

ARG PG_MAJOR
ARG ONNX_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PG_MAJOR=${PG_MAJOR} \
    ONNX_VERSION=${ONNX_VERSION} \
    ONNX_PATH=/usr/local/onnxruntime \
    PG_CONFIG=/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        clang \
        cmake \
        make \
        git \
        curl \
        wget \
        ca-certificates \
        pkg-config \
        libssl-dev \
        libcurl4-openssl-dev \
        zlib1g-dev \
        python3 \
        python3-pip \
        postgresql-server-dev-${PG_MAJOR} && \
    rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime (CPU build) matching architecture
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
        amd64) onnx_pkg="onnxruntime-linux-x64-${ONNX_VERSION}.tgz" ;; \
        arm64) onnx_pkg="onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz" ;; \
        *) echo "Unsupported architecture for ONNX Runtime: ${arch}" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${onnx_pkg}" -o /tmp/onnxruntime.tgz; \
    mkdir -p /tmp/onnxruntime; \
    tar -xzf /tmp/onnxruntime.tgz -C /tmp/onnxruntime --strip-components=1; \
    mv /tmp/onnxruntime ${ONNX_PATH}; \
    rm -f /tmp/onnxruntime.tgz

WORKDIR /build/Neurondb

COPY . .

# Build and install NeuronDB into the builder image
# Note: Architecture is automatically detected and handled by the build system
RUN make clean && \
    make && \
    make install

# -----------------------------------------------------------------------------

FROM postgres:${PG_MAJOR}-bookworm

ARG PG_MAJOR
ARG ONNX_VERSION

ENV ONNX_PATH=/usr/local/onnxruntime \
    NEURONDB_HOME=/usr/local/share/neurondb \
    LANG=C.UTF-8

RUN mkdir -p ${NEURONDB_HOME} /docker-entrypoint-initdb.d

# Runtime dependencies required by neurondb.so
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcurl4 \
        libssl3 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy runtime assets from builder
COPY --from=builder /usr/lib/postgresql/${PG_MAJOR}/lib/neurondb*.so /usr/lib/postgresql/${PG_MAJOR}/lib/
COPY --from=builder /usr/share/postgresql/${PG_MAJOR}/extension/neurondb* /usr/share/postgresql/${PG_MAJOR}/extension/
COPY --from=builder /usr/local/onnxruntime ${ONNX_PATH}

# Ship helper init scripts
COPY docker/docker-entrypoint-initdb.d/ /docker-entrypoint-initdb.d/

VOLUME ["/var/lib/postgresql/data"]

HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD pg_isready -U "${POSTGRES_USER:-postgres}"

# Default entrypoint inherited from postgres image

