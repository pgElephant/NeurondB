#!/usr/bin/env bash
set -euo pipefail

# Purpose:
# - Install minimal prerequisites per platform
# - Select platform Makefile and wire it by copying to Makefile
# - Do not build; just prep environment and Makefile
#
# Usage:
#   ./build.sh
#

msg() { printf "[BUILD] %s\n" "$*"; }
err() { printf "[ERROR] %s\n" "$*" >&2; }
ts() { date "+%Y-%m-%d %H:%M:%S"; }

VERBOSE=0
# Ensure GPU_MODE has a safe default even under 'set -u'
GPU_MODE="${GPU_MODE:-none}"
if [[ "${1:-}" == "-v" || "${1:-}" == "--verbose" ]]; then
	VERBOSE=1
	shift || true
fi

run_step() {
	# usage: run_step "message" cmd...
	local message="$1"; shift
	local start elapsed rc
	# colors and glyphs
	local GREEN="\033[0;32m"
	local RED="\033[0;31m"
	local YELLOW="\033[1;33m"
	local NC="\033[0m"
	local OK="✓" FAIL="✗"
	start=$(date +%s)
	if [[ $VERBOSE -eq 1 ]]; then
		printf "[%s] %s\n" "$(ts)" "$message"
		"$@"
		rc=$?
	else
		# capture output quietly
		if output=$("$@" 2>&1); then
			rc=0
		else
			rc=$?
		fi
	fi
	elapsed=$(( $(date +%s) - start ))
	if [[ $rc -eq 0 ]]; then
		# aligned: timestamp (19), message (fixed 40), elapsed (right, 6 incl. 's')
		printf "${GREEN}%s${NC} %s  %-40s %6ds\n" "$OK" "$(ts)" "$message" "$elapsed"
	else
		printf "${RED}%s${NC} %s  %-40s %6ds\n" "$FAIL" "$(ts)" "$message" "$elapsed"
		[[ $VERBOSE -eq 0 ]] && { printf "%s\n" "$output" >&2 || true; }
		exit $rc
	fi
}

# Print detailed OS information
print_os_details() {
	if [[ -f /etc/os-release ]]; then
		. /etc/os-release
		printf '%s\n' "------------------------------------------------------------------------------------"
		printf '\033[1m%-16s\033[0m : %s\n' "OS PRETTY_NAME" "${PRETTY_NAME:-unknown}"
		printf '\033[1m%-16s\033[0m : %s\n' "OS ID" "${ID:-unknown}"
		printf '\033[1m%-16s\033[0m : %s\n' "OS VERSION_ID" "${VERSION_ID:-unknown}"
	else
		printf '%s\n' "------------------------------------------------------------------------------------"
		printf '\033[1m%-16s\033[0m : %s\n' "OS PRETTY_NAME" "$(uname -s)"
		printf '\033[1m%-16s\033[0m : %s\n' "OS ID" "unknown"
		printf '\033[1m%-16s\033[0m : %s\n' "OS VERSION_ID" "unknown"
	fi
	printf '\033[1m%-16s\033[0m : %s\n' "Kernel" "$(uname -r)"
	printf '\033[1m%-16s\033[0m : %s\n' "Architecture" "$(uname -m)"
	printf '\033[1m%-16s\033[0m : %s\n' "Hostname" "$(hostname)"
	printf '%s\n' "------------------------------------------------------------------------------------"
}

# Print detailed PostgreSQL information via pg_config
print_pg_details() {
	local pgc
	pgc="${SELECTED_PG_CONFIG:-$(command -v pg_config || true)}"
	if [[ -z "$pgc" ]]; then
		printf '%s\n' "pg_config not found"
		return 1
	fi
	printf '%s\n' "------------------------------------------------------------------------------------"
	printf '\033[1m%-16s\033[0m : %s\n' "pg_config" "$pgc"
	printf '\033[1m%-16s\033[0m : %s\n' "Version" "$($pgc --version 2>/dev/null)"
	printf '\033[1m%-16s\033[0m : %s\n' "BINDIR" "$($pgc --bindir 2>/dev/null)"
	printf '\033[1m%-16s\033[0m : %s\n' "PKGLIBDIR" "$($pgc --pkglibdir 2>/dev/null)"
	printf '\033[1m%-16s\033[0m : %s\n' "SHAREDIR" "$($pgc --sharedir 2>/dev/null)"
	printf '\033[1m%-16s\033[0m : %s\n' "INCLUDEDIR" "$($pgc --includedir 2>/dev/null)"
	printf '\033[1m%-16s\033[0m : %s\n' "INCLUDEDIR-S" "$($pgc --includedir-server 2>/dev/null)"
	printf '\033[1m%-16s\033[0m : %s\n' "LIBDIR" "$($pgc --libdir 2>/dev/null)"
	printf '%s\n' "------------------------------------------------------------------------------------"
	return 0
}

# Detect PostgreSQL major version to guide package selection
PG_MAJOR_SELECTED=""
detect_pg_major() {
	# Try pg_config first
	if command -v pg_config >/dev/null 2>&1; then
		local ver
		ver="$(pg_config --version 2>/dev/null | awk '{print $2}')"
		PG_MAJOR_SELECTED="$(echo "$ver" | cut -d. -f1)"
		return 0
	fi
	# Fallbacks per platform
	local uname_s
	uname_s="$(uname -s)"
	if [[ "$uname_s" == "Linux" ]]; then
		# Try to detect highest available dev package on Debian/Ubuntu
		if command -v apt-cache >/dev/null 2>&1; then
			local cand
			cand="$(apt-cache search -n '^postgresql-server-dev-[0-9]+$' 2>/dev/null | awk '{print $1}' | sed -n 's/^postgresql-server-dev-\([0-9][0-9]*\)$/\1/p' | sort -nr | head -1)"
			if [[ -n "$cand" ]]; then
				PG_MAJOR_SELECTED="$cand"
				return 0
			fi
		fi
		# Try RPM-based listing
		if command -v dnf >/dev/null 2>&1; then
			local cand
			cand="$(dnf list --available 'postgresql*-devel' 2>/dev/null | sed -n 's/^postgresql\([0-9][0-9]*\)-devel\..*/\1/p' | sort -nr | head -1)"
			if [[ -n "$cand" ]]; then
				PG_MAJOR_SELECTED="$cand"
				return 0
			fi
		fi
	fi
	if [[ "$uname_s" == "Darwin" ]]; then
		# On macOS, default to 18 if not found
		PG_MAJOR_SELECTED="18"
		return 0
	fi
	# Generic default
	PG_MAJOR_SELECTED="18"
	return 0
}

# Print detailed GPU information (CUDA/ROCm/Metal)
print_gpu_details() {
	printf '%s\n' "------------------------------------------------------------------------------------"
	# CUDA detection
	local nvcc_path cuda_home cuda_ver nvsmi
	nvcc_path="$(command -v nvcc || true)"
	if [[ -n "$nvcc_path" ]]; then
		cuda_home="$(dirname "$(dirname "$nvcc_path")")"
		cuda_ver="$("$nvcc_path" --version 2>/dev/null | awk -F'release ' '/release/ {print $2}' | cut -d, -f1 | head -n1)"
		printf '\033[1m%-16s\033[0m : %s\n' "CUDA" "Yes"
		printf '\033[1m%-16s\033[0m : %s\n' "CUDA_HOME" "$cuda_home"
		printf '\033[1m%-16s\033[0m : %s\n' "CUDA Version" "${cuda_ver:-unknown}"
	else
		printf '\033[1m%-16s\033[0m : %s\n' "CUDA" "No"
	fi
	# NVIDIA driver/runtime
	nvsmi="$(command -v nvidia-smi || true)"
	if [[ -n "$nvsmi" ]]; then
		local driver line
		line="$("$nvsmi" --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n1)"
		driver="$(echo "$line" | awk -F, '{gsub(/^ +| +$/,"",$2); print $2}')"
		printf '\033[1m%-16s\033[0m : %s\n' "NVIDIA-SMI" "$nvsmi"
		[[ -n "$line" ]] && printf '\033[1m%-16s\033[0m : %s\n' "GPU/Driver" "$line"
		[[ -n "$driver" ]] || true
	fi
	# ROCm detection
	local hipcc_path rocm_home rocm_ver
	hipcc_path="$(command -v hipcc || true)"
	if [[ -n "$hipcc_path" ]]; then
		rocm_home="$(dirname "$(dirname "$hipcc_path")")"
		if [[ -f "$rocm_home/.info/version" ]]; then
			rocm_ver="$(cat "$rocm_home/.info/version" 2>/dev/null)"
		fi
		printf '\033[1m%-16s\033[0m : %s\n' "ROCm" "Yes"
		printf '\033[1m%-16s\033[0m : %s\n' "ROCM_PATH" "$rocm_home"
		printf '\033[1m%-16s\033[0m : %s\n' "ROCm Version" "${rocm_ver:-unknown}"
	else
		printf '\033[1m%-16s\033[0m : %s\n' "ROCm" "No"
	fi
	# Metal (macOS)
	if [[ "$(uname -s)" == "Darwin" ]]; then
		# Assume Metal available on macOS; check for Xcode CLT
		local xcode_clt
		if xcode_clt="$(xcode-select -p 2>/dev/null)"; then
			printf '\033[1m%-16s\033[0m : %s\n' "Metal" "Yes"
			printf '\033[1m%-16s\033[0m : %s\n' "Xcode CLT" "$xcode_clt"
		else
			printf '\033[1m%-16s\033[0m : %s\n' "Metal" "Unknown (install Xcode CLT)"
		fi
	else
		printf '\033[1m%-16s\033[0m : %s\n' "Metal" "No"
	fi
	printf '%s\n' "------------------------------------------------------------------------------------"
	return 0
}

select_pg_config() {
	# Prefer explicit PG_CONFIG, then system pg_config, then common local prefixes
	if [[ -n "${PG_CONFIG:-}" && -x "${PG_CONFIG:-}" ]]; then
		SELECTED_PG_CONFIG="$PG_CONFIG"
		export SELECTED_PG_CONFIG
		return 0
	fi
	if command -v pg_config >/dev/null 2>&1; then
		SELECTED_PG_CONFIG="$(command -v pg_config)"
		export SELECTED_PG_CONFIG
		return 0
	fi
	local candidate
	for candidate in /usr/local/pgsql.18/bin/pg_config /usr/local/pgsql/bin/pg_config; do
		if [[ -x "$candidate" ]]; then
			SELECTED_PG_CONFIG="$candidate"
			export SELECTED_PG_CONFIG
			return 0
		fi
	done
	SELECTED_PG_CONFIG=""
	return 1
}

detect_gpu_paths() {
	# Set CUDA_PATH and GPU_MODE based on system detection
	if command -v nvcc >/dev/null 2>&1; then
		local nvcc_path cuda_home
		nvcc_path="$(command -v nvcc)"
		cuda_home="$(dirname "$(dirname "$nvcc_path")")"
		if [[ -f "$cuda_home/include/cuda_runtime.h" ]]; then
			CUDA_PATH="$cuda_home"
			GPU_MODE="cuda"
		fi
	fi
	# ROCm
	if [[ -z "${GPU_MODE:-}" || "$GPU_MODE" == "none" ]]; then
		local hipcc_path rocm_home
		hipcc_path="$(command -v hipcc || true)"
		if [[ -n "$hipcc_path" ]]; then
			rocm_home="$(dirname "$(dirname "$hipcc_path")")"
			if [[ -f "$rocm_home/include/hip/hip_runtime.h" ]]; then
				ROCM_PATH="$rocm_home"
				GPU_MODE="rocm"
			fi
		fi
	fi
	# Default when nothing found
	GPU_MODE="${GPU_MODE:-none}"
}

run_step_stream() {
	# usage: run_step_stream "message" cmd...
	local message="$1"; shift
	local start elapsed rc
	local GREEN="\033[0;32m"
	local RED="\033[0;31m"
	local NC="\033[0m"
	local OK="✓" FAIL="✗"
	start=$(date +%s)
	printf "[%s] %s\n" "$(ts)" "$message"
	"$@"
	rc=$?
	elapsed=$(( $(date +%s) - start ))
	if [[ $rc -eq 0 ]]; then
		printf "${GREEN}%s${NC} %s  %-40s %6ds\n" "$OK" "$(ts)" "$message" "$elapsed"
	else
		printf "${RED}%s${NC} %s  %-40s %6ds\n" "$FAIL" "$(ts)" "$message" "$elapsed"
		exit $rc
	fi
}

detect_platform() {
	if [[ "$(uname -s)" == "Darwin" ]]; then
		echo "osx"
		return
	fi
	if [[ -f /etc/rocky-release ]] || grep -qiE 'rocky|centos|rhel' /etc/os-release 2>/dev/null; then
		echo "rocky"
		return
	fi
	if command -v apt-get >/dev/null 2>&1 || grep -qiE 'debian|ubuntu' /etc/os-release 2>/dev/null; then
		echo "deb"
		return
	fi
	echo "unknown"
}

install_prereqs_deb() {
	msg "Installing prerequisites for Debian/Ubuntu (requires sudo)..."
	local pgdev_pkg="postgresql-server-dev-${PG_MAJOR_SELECTED:-18}"
	echo "Packages: build-essential git curl ca-certificates pkg-config libxml2-dev libssl-dev libcurl4-openssl-dev $pgdev_pkg"
	if [[ $VERBOSE -eq 1 ]]; then
		sudo apt-get update -y
		sudo apt-get install -y --no-install-recommends \
			build-essential git curl ca-certificates pkg-config \
			libxml2-dev libssl-dev libcurl4-openssl-dev \
			"$pgdev_pkg"
	else
		sudo apt-get update -y >/dev/null
		sudo apt-get install -y --no-install-recommends \
			build-essential git curl ca-certificates pkg-config \
			libxml2-dev libssl-dev libcurl4-openssl-dev \
			"$pgdev_pkg" >/dev/null
	fi
}

install_prereqs_rocky() {
	msg "Installing prerequisites for Rocky/RHEL (requires sudo)..."
	if command -v dnf >/dev/null 2>&1; then
		PKG=dnf
	else
		PKG=yum
	fi
	local pgdev_pkg="postgresql${PG_MAJOR_SELECTED:-18}-devel"
	echo "Packages: gcc gcc-c++ make git curl ca-certificates pkgconfig libxml2-devel openssl-devel libcurl-devel [$pgdev_pkg if available]"
	if [[ $VERBOSE -eq 1 ]]; then
		sudo $PKG install -y \
			gcc gcc-c++ make git curl ca-certificates pkgconfig \
			libxml2-devel openssl-devel libcurl-devel
	else
		sudo $PKG install -y \
			gcc gcc-c++ make git curl ca-certificates pkgconfig \
			libxml2-devel openssl-devel libcurl-devel >/dev/null
	fi
	# PostgreSQL devel (tries common module streams; adjust if needed)
	if ! command -v pg_config >/dev/null 2>&1; then
		# Prefer versioned devel; fallback to generic
		if [[ $VERBOSE -eq 1 ]]; then
			sudo $PKG install -y "$pgdev_pkg" || sudo $PKG install -y postgresql-devel || true
		else
			sudo $PKG install -y "$pgdev_pkg" >/dev/null || sudo $PKG install -y postgresql-devel >/dev/null || true
		fi
	fi
}

install_prereqs_osx() {
	msg "Installing prerequisites for macOS via Homebrew..."
	if ! command -v brew >/dev/null 2>&1; then
		err "Homebrew is required. Install from https://brew.sh and re-run."
		exit 1
	fi
	local pg_formula="postgresql@${PG_MAJOR_SELECTED:-18}"
	echo "Packages: llvm make git curl pkg-config libxml2 openssl@3 $pg_formula"
	if [[ $VERBOSE -eq 1 ]]; then
		brew update
		# Install PostgreSQL versioned formula if not already present
		if ! brew list --versions "$pg_formula" >/dev/null 2>&1; then
			brew install "$pg_formula" || true
		fi
		# Core tools
		brew install \
			llvm make git curl pkg-config libxml2 openssl@3
	else
		brew update >/dev/null
		if ! brew list --versions "$pg_formula" >/dev/null 2>&1; then
			brew install "$pg_formula" >/dev/null || true
		fi
		brew install \
			llvm make git curl pkg-config libxml2 openssl@3 >/dev/null
	fi
	# Ensure pg_config is on PATH
	if ! command -v pg_config >/dev/null 2>&1; then
		PGBIN="$(brew --prefix "$pg_formula" 2>/dev/null)/bin"
		if [[ -x "$PGBIN/pg_config" ]]; then
			msg "PostgreSQL found at $PGBIN. Add to PATH in your shell profile."
		fi
	fi
}

wire_makefile() {
	local platform="$1"
	local mf_src
	case "$platform" in
		deb)   mf_src="Makefile.deb" ;;
		rocky) mf_src="Makefile.rocky" ;;
		osx)   mf_src="Makefile.osx" ;;
		*)     err "Unsupported platform key: $platform"; exit 1 ;;
	esac
	if [[ ! -f "$mf_src" ]]; then
		err "Missing $mf_src next to build.sh"
		exit 1
	fi
	# Preserve existing Makefile once as Makefile.core
	if [[ -f Makefile && ! -f Makefile.core ]]; then
		msg "Preserving existing Makefile -> Makefile.core"
		mv Makefile Makefile.core
	fi
	msg "Copy $mf_src -> Makefile"
	# Announce which Makefile is being created from which OS wrapper
	printf "Creating Makefiles [%s]\n" "$mf_src"
	cp -f "$mf_src" Makefile
}

write_makefile_local() {
	# Persist detected settings for Makefile consumption
	local pgc="${SELECTED_PG_CONFIG:-$(command -v pg_config || true)}"
	local gm="${GPU_MODE:-none}"
	local cp="${CUDA_PATH:-${CUDA_HOME:-}}"
	cat > Makefile.local <<EOF
# Autogenerated by build.sh - do not edit manually
PG_MAJOR ?= ${PG_MAJOR_SELECTED}
PG_CONFIG ?= ${pgc}
GPU_MODE ?= ${gm}
CUDA_PATH ?= ${cp}
QUIET_ONNX ?= 1
V ?= 1
# Avoid LTO crashes with PGXS/llvm-lto on some toolchains
CFLAGS += -O2 -fPIC -fno-lto
LDFLAGS += -fno-lto
.PHONY: compile
.DEFAULT_GOAL := compile
compile: all
EOF
}

main() {
	cd "$(dirname "$0")"
	local platform
	platform="$(detect_platform)"
	if [[ $VERBOSE -eq 1 ]]; then
		printf "[%s] Detected platform: %s\n" "$(ts)" "$platform"
	fi
	if [[ "$platform" == "unknown" ]]; then
		# OS detection failed
		printf "\033[0;31m%s\033[0m %s  %-40s %6ds\n" "✗" "$(ts)" "Detect OS details" 0
		exit 1
	fi
	# Show detailed OS detection (single completion line)
	print_os_details
	printf "\033[0;32m%s\033[0m %s  %-40s %6ds\n" "✓" "$(ts)" "Detect OS details" 0
	# Show detailed GPU detection (single completion line)
	print_gpu_details
	printf "\033[0;32m%s\033[0m %s  %-40s %6ds\n" "✓" "$(ts)" "Detect GPU details" 0
	# Export detected GPU toolchain paths to Makefiles
	detect_gpu_paths
	# Detect PostgreSQL version to guide prerequisites
	select_pg_config || true
	detect_pg_major
	if command -v pg_config >/dev/null 2>&1; then
		# Show detailed PostgreSQL detection (single completion line)
		print_pg_details
		printf "\033[0;32m%s\033[0m %s  %-40s %6ds\n" "✓" "$(ts)" "Detect PostgreSQL details" 0
	else
		# Still proceed with PG_MAJOR_SELECTED for package selection, but fail if later pg_config remains missing
		printf "\033[1;33m%s\033[0m %s  %-40s %6ds\n" "⏭" "$(ts)" "pg_config not found; will install dev packages for PG ${PG_MAJOR_SELECTED}" 0
	fi
	if [[ $VERBOSE -eq 1 ]]; then
		case "$platform" in
			deb)   run_step_stream "Install prerequisites (Debian/Ubuntu)" install_prereqs_deb ;;
			rocky) run_step_stream "Install prerequisites (Rocky/RHEL)"   install_prereqs_rocky ;;
			osx)   run_step_stream "Install prerequisites (macOS)"        install_prereqs_osx ;;
			*)     printf "✗ %s  Unsupported/unknown platform  0s\n" "$(ts)"; exit 1 ;;
		esac
	else
		case "$platform" in
			deb)   run_step "Install prerequisites (Debian/Ubuntu)" install_prereqs_deb ;;
			rocky) run_step "Install prerequisites (Rocky/RHEL)"   install_prereqs_rocky ;;
			osx)   run_step "Install prerequisites (macOS)"        install_prereqs_osx ;;
			*)     printf "✗ %s  Unsupported/unknown platform  0s\n" "$(ts)"; exit 1 ;;
		esac
	fi
	# Persist detections for Makefiles
	run_step "Write Makefile.local" write_makefile_local
	run_step "Select platform Makefile ($platform)" wire_makefile "$platform"
	# Ensure pg_config exists after install; otherwise, error
	if ! command -v pg_config >/dev/null 2>&1; then
		printf "\033[0;31m%s\033[0m %s  %-40s %6ds\n" "✗" "$(ts)" "PostgreSQL dev tools not found after install" 0
		exit 1
	fi
	# final summary line
	printf "\033[0;32m%s\033[0m %s  %-40s %6ds\n" "✓" "$(ts)" "Prep complete" 0
}

main "$@"
exit 0

#!/bin/bash
#-------------------------------------------------------------------------
#
# build.sh
#     Platform-aware build script for NeurondB
#
# Automatically detects OS and PostgreSQL version, installs dependencies,
# and builds NeurondB with optional GPU support.
#
# Copyright (c) 2024-2025, pgElephant, Inc. <admin@pgelephant.com>
#
#-------------------------------------------------------------------------

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect platform
detect_platform() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
    elif [ "$(uname)" = "Darwin" ]; then
        OS="macos"
        VERSION=$(sw_vers -productVersion)
    else
        log_error "Unsupported operating system"
        exit 1
    fi
    
    log_info "Detected platform: $OS $VERSION"
}

# Detect PostgreSQL version
detect_postgres_version() {
    if command -v pg_config >/dev/null 2>&1; then
        PG_VERSION=$(pg_config --version | awk '{print $2}' | cut -d. -f1)
        PG_CONFIG_PATH=$(command -v pg_config)
        log_info "Found PostgreSQL $PG_VERSION at $PG_CONFIG_PATH"
    else
        log_warning "pg_config not found, will install PostgreSQL"
        PG_VERSION=${PG_VERSION:-17}
    fi
}

# Install dependencies on Ubuntu/Debian
install_ubuntu_deps() {
    log_info "Installing dependencies on Ubuntu/Debian..."
    
    SUDO=""
    if [ "$(id -u)" != "0" ]; then
        SUDO="sudo"
    fi
    
    $SUDO apt-get update
    
    # Build essentials
    $SUDO apt-get install -y \
        build-essential \
        git \
        wget \
        curl \
        pkg-config \
        libcurl4-openssl-dev \
        libssl-dev \
        zlib1g-dev
    
    # PostgreSQL
    if ! command -v pg_config >/dev/null 2>&1; then
        log_info "Installing PostgreSQL $PG_VERSION..."
        $SUDO apt-get install -y \
            postgresql-$PG_VERSION \
            postgresql-server-dev-$PG_VERSION \
            postgresql-contrib-$PG_VERSION
    else
        log_info "PostgreSQL already installed"
        $SUDO apt-get install -y \
            postgresql-server-dev-$PG_VERSION
    fi
    
    # ONNX Runtime for HuggingFace model support
    log_info "Installing ONNX Runtime..."
    ONNX_VERSION="1.17.0"
    ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
    
    if [ ! -d "/usr/local/onnxruntime" ]; then
        wget -q $ONNX_URL -O /tmp/onnxruntime.tgz
        $SUDO mkdir -p /usr/local/onnxruntime
        $SUDO tar -xzf /tmp/onnxruntime.tgz -C /usr/local/onnxruntime --strip-components=1
        rm /tmp/onnxruntime.tgz
        log_success "ONNX Runtime installed to /usr/local/onnxruntime"
    else
        log_info "ONNX Runtime already installed"
    fi
    
    # Optional: GPU libraries
    if [ "$WITH_GPU" = "yes" ]; then
        log_info "Checking for CUDA..."
        for path in /usr/local/cuda /opt/cuda /usr/cuda; do
            if [ -d "$path/include" ] && [ -f "$path/include/cuda_runtime.h" ]; then
                log_success "CUDA found at $path"
                CUDA_PATH="$path"
                break
            fi
        done
        if [ -z "$CUDA_PATH" ]; then
            log_warning "CUDA toolkit not found. Attempting to install nvidia-cuda-toolkit via apt."
            if $SUDO apt-get install -y nvidia-cuda-toolkit; then
                for path in /usr/local/cuda /usr/lib/cuda; do
                    if [ -d "$path/include" ] && [ -f "$path/include/cuda_runtime.h" ]; then
                        CUDA_PATH="$path"
                        log_success "CUDA toolkit installed at $CUDA_PATH"
                        break
                    fi
                done
                if [ -z "$CUDA_PATH" ]; then
                    log_warning "Installed nvidia-cuda-toolkit but cuda_runtime.h still not found"
                fi
            else
                log_warning "Automatic CUDA installation failed. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
            fi
        fi
        
        if [ -z "$CUDA_PATH" ]; then
            log_warning "CUDA not available. NeurondB will build without GPU kernels."
        fi
        
        # Install ONNX Runtime with CUDA support
        if [ -n "$CUDA_PATH" ]; then
            log_info "Installing ONNX Runtime with CUDA support..."
            ONNX_CUDA_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz"
            wget -q $ONNX_CUDA_URL -O /tmp/onnxruntime-cuda.tgz
            $SUDO tar -xzf /tmp/onnxruntime-cuda.tgz -C /usr/local/onnxruntime --strip-components=1
            rm /tmp/onnxruntime-cuda.tgz
            log_success "ONNX Runtime with CUDA installed"
        fi
    fi
    
    log_success "Ubuntu dependencies installed"
}

# Install dependencies on Rocky Linux/RHEL/CentOS
install_rocky_deps() {
    log_info "Installing dependencies on Rocky Linux/RHEL..."
    
    SUDO=""
    if [ "$(id -u)" != "0" ]; then
        SUDO="sudo"
    fi
    
    # Enable PostgreSQL repository
    if [ ! -f /etc/yum.repos.d/pgdg-redhat-all.repo ]; then
        log_info "Installing PostgreSQL repository..."
        $SUDO dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-$(rpm -E %{rhel})-x86_64/pgdg-redhat-repo-latest.noarch.rpm
    fi
    
    # Disable built-in PostgreSQL module
    $SUDO dnf -qy module disable postgresql
    
    # Build essentials
    $SUDO dnf install -y \
        gcc \
        gcc-c++ \
        make \
        git \
        wget \
        curl \
        redhat-rpm-config \
        libcurl-devel \
        openssl-devel \
        zlib-devel \
        pkg-config
    
    # PostgreSQL
    if ! command -v pg_config >/dev/null 2>&1; then
        log_info "Installing PostgreSQL $PG_VERSION..."
        $SUDO dnf install -y \
            postgresql$PG_VERSION \
            postgresql$PG_VERSION-server \
            postgresql$PG_VERSION-devel \
            postgresql$PG_VERSION-contrib
    else
        log_info "PostgreSQL already installed"
        $SUDO dnf install -y \
            postgresql$PG_VERSION-devel
    fi
    
    # ONNX Runtime for HuggingFace model support
    log_info "Installing ONNX Runtime..."
    ONNX_VERSION="1.17.0"
    ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
    
    if [ ! -d "/usr/local/onnxruntime" ]; then
        wget -q $ONNX_URL -O /tmp/onnxruntime.tgz
        $SUDO mkdir -p /usr/local/onnxruntime
        $SUDO tar -xzf /tmp/onnxruntime.tgz -C /usr/local/onnxruntime --strip-components=1
        rm /tmp/onnxruntime.tgz
        log_success "ONNX Runtime installed to /usr/local/onnxruntime"
    else
        log_info "ONNX Runtime already installed"
    fi
    
    # Optional: GPU libraries
    if [ "$WITH_GPU" = "yes" ]; then
        log_info "Checking for ROCm..."
        for path in /opt/rocm /usr/rocm; do
            if [ -d "$path/include" ] && [ -f "$path/include/hip/hip_runtime.h" ]; then
                log_success "ROCm found at $path"
                ROCM_PATH="$path"
                break
            fi
        done
        if [ -z "$ROCM_PATH" ]; then
            log_warning "ROCm not found. Install from: https://rocm.docs.amd.com/"
            log_warning "NeurondB will build without GPU support"
        fi
        
        # Install ONNX Runtime with ROCm/CUDA support if available
        if [ -n "$ROCM_PATH" ] || [ -n "$CUDA_PATH" ]; then
            log_info "Installing ONNX Runtime with GPU support..."
            ONNX_GPU_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz"
            wget -q $ONNX_GPU_URL -O /tmp/onnxruntime-gpu.tgz
            $SUDO tar -xzf /tmp/onnxruntime-gpu.tgz -C /usr/local/onnxruntime --strip-components=1
            rm /tmp/onnxruntime-gpu.tgz
            log_success "ONNX Runtime with GPU support installed"
        fi
    fi
    
    log_success "Rocky Linux dependencies installed"
}

# Install dependencies on macOS
install_macos_deps() {
    log_info "Installing dependencies on macOS..."
    
    # Check for Homebrew
    if ! command -v brew >/dev/null 2>&1; then
        log_error "Homebrew not found. Install from: https://brew.sh"
        exit 1
    fi
    
    # Build essentials
    brew install \
        gcc \
        make \
        git \
        wget \
        curl \
        pkg-config \
        openssl \
        zlib
    
    # PostgreSQL
    if ! command -v pg_config >/dev/null 2>&1; then
        log_info "Installing PostgreSQL $PG_VERSION..."
        brew install postgresql@$PG_VERSION
    else
        log_info "PostgreSQL already installed"
    fi
    
    # ONNX Runtime for HuggingFace model support
    log_info "Installing ONNX Runtime for macOS..."
    ONNX_VERSION="1.17.0"
    
    # Detect macOS architecture (Intel vs Apple Silicon)
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-osx-arm64-${ONNX_VERSION}.tgz"
        log_info "Detected Apple Silicon (arm64)"
    else
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-osx-x86_64-${ONNX_VERSION}.tgz"
        log_info "Detected Intel (x86_64)"
    fi
    
    if [ ! -d "/usr/local/onnxruntime" ]; then
        if command -v curl >/dev/null 2>&1; then
            curl -L -o /tmp/onnxruntime.tgz $ONNX_URL
        else
            wget -q $ONNX_URL -O /tmp/onnxruntime.tgz
        fi
        sudo mkdir -p /usr/local/onnxruntime
        sudo tar -xzf /tmp/onnxruntime.tgz -C /usr/local/onnxruntime --strip-components=1
        rm /tmp/onnxruntime.tgz
        log_success "ONNX Runtime installed to /usr/local/onnxruntime"
    else
        log_info "ONNX Runtime already installed"
    fi
    
    # Optional: GPU libraries (CUDA for macOS if available)
    if [ "$WITH_GPU" = "yes" ]; then
        log_info "Checking for CUDA on macOS..."
        for path in /usr/local/cuda /opt/cuda; do
            if [ -d "$path/include" ] && [ -f "$path/include/cuda_runtime.h" ]; then
                log_success "CUDA found at $path"
                CUDA_PATH="$path"
                break
            fi
        done
        if [ -z "$CUDA_PATH" ]; then
            log_warning "GPU support on macOS requires CUDA toolkit"
            log_warning "Download from: https://developer.nvidia.com/cuda-downloads"
            log_warning "For Apple Silicon, ONNX Runtime uses CoreML acceleration"
            log_warning "NeurondB will build without CUDA GPU support"
        fi
    fi
    
    log_success "macOS dependencies installed"
}

detect_onnx_runtime() {
    local header="include/onnxruntime/core/session/onnxruntime_cxx_api.h"
    if [ -n "$ONNX_PATH" ] && [ -f "$ONNX_PATH/$header" ]; then
        if [ -z "$ONNX_RUNTIME_PATH" ]; then
            ONNX_RUNTIME_PATH="$ONNX_PATH"
        fi
        log_info "Detected ONNX Runtime at $ONNX_PATH"
        return
    fi
    if [ -n "$ONNX_RUNTIME_PATH" ] && [ -f "$ONNX_RUNTIME_PATH/$header" ]; then
        if [ -z "$ONNX_PATH" ]; then
            ONNX_PATH="$ONNX_RUNTIME_PATH"
        fi
        log_info "Detected ONNX Runtime at $ONNX_RUNTIME_PATH"
        return
    fi

    local candidates=(
        "$HOME/onnxruntime-gpu"
        "$HOME/onnxruntime"
        "/usr/local/onnxruntime"
        "/usr/lib/onnxruntime"
        "/opt/onnxruntime"
    )

    for path in "${candidates[@]}"; do
        if [ -n "$path" ] && [ -f "$path/$header" ]; then
            ONNX_PATH="$path"
            ONNX_RUNTIME_PATH="${ONNX_RUNTIME_PATH:-$path}"
            log_info "Detected ONNX Runtime at $path"
            return
        fi
    done

    log_info "ONNX Runtime not found automatically. Set --onnx-path or ONNX_PATH if required."
}

detect_cuda_toolkit() {
    if [ "$WITH_GPU" = "no" ]; then
        return
    fi

    if [ -n "$CUDA_PATH" ] && [ -f "$CUDA_PATH/include/cuda_runtime.h" ]; then
        log_info "Using CUDA toolkit at $CUDA_PATH"
        return
    fi

    if command -v nvcc >/dev/null 2>&1; then
        local nvcc_path
        nvcc_path=$(command -v nvcc)
        local nvcc_root
        nvcc_root=$(dirname "$(dirname "$nvcc_path")")
        if [ -f "$nvcc_root/include/cuda_runtime.h" ]; then
            CUDA_PATH="$nvcc_root"
            log_info "Detected CUDA via nvcc at $CUDA_PATH"
            return
        fi
    fi

    local candidates=(
        "$CUDA_HOME"
        "/usr/local/cuda"
        "/opt/cuda"
        "/usr/lib/cuda"
        "/usr"
    )

    for path in "${candidates[@]}"; do
        if [ -n "$path" ] && [ -f "$path/include/cuda_runtime.h" ]; then
            CUDA_PATH="$path"
            log_info "Detected CUDA toolkit at $CUDA_PATH"
            return
        fi
    done

    if [ "$WITH_GPU" = "yes" ]; then
        log_warning "CUDA toolkit headers not found. GPU build will fall back to CPU unless CUDA_PATH is provided."
    fi
}

detect_rocm_toolkit() {
    if [ "$WITH_GPU" = "no" ]; then
        return
    fi

    if [ -n "$ROCM_PATH" ] && [ -f "$ROCM_PATH/include/hip/hip_runtime.h" ]; then
        log_info "Using ROCm toolkit at $ROCM_PATH"
        return
    fi

    local candidates=(
        "${ROCM_HOME:-}"
        "/opt/rocm"
        "/usr/rocm"
    )

    for path in "${candidates[@]}"; do
        if [ -n "$path" ] && [ -f "$path/include/hip/hip_runtime.h" ]; then
            ROCM_PATH="$path"
            log_info "Detected ROCm toolkit at $ROCM_PATH"
            return
        fi
    done
}

prepare_make_args() {
    local log_mode="${1:-log}"
    MAKE_ARGS=""

    if [ -n "$ONNX_PATH" ]; then
        MAKE_ARGS="$MAKE_ARGS ONNX_PATH=$ONNX_PATH"
        if [ "$log_mode" = "log" ]; then
            log_info "Using ONNX Runtime headers at: $ONNX_PATH"
        fi
    fi
    if [ -n "$ONNX_RUNTIME_PATH" ]; then
        MAKE_ARGS="$MAKE_ARGS ONNX_RUNTIME_PATH=$ONNX_RUNTIME_PATH"
        if [ "$log_mode" = "log" ]; then
            log_info "Using ONNX Runtime libraries at: $ONNX_RUNTIME_PATH"
        fi
    fi

    if [ "$WITH_GPU" = "yes" ]; then
        if [ -n "$CUDA_PATH" ]; then
            MAKE_ARGS="$MAKE_ARGS CUDA_PATH=$CUDA_PATH"
            if [ "$log_mode" = "log" ]; then
                log_info "Building with CUDA support from: $CUDA_PATH"
            fi
        fi
        if [ -n "$ROCM_PATH" ]; then
            MAKE_ARGS="$MAKE_ARGS ROCM_PATH=$ROCM_PATH"
            if [ "$log_mode" = "log" ]; then
                log_info "Building with ROCm support from: $ROCM_PATH"
            fi
        fi
        if [ -z "$CUDA_PATH" ] && [ -z "$ROCM_PATH" ] && [ "$log_mode" = "log" ]; then
            log_warning "GPU support requested but no CUDA/ROCm toolkit detected. CPU fallback will remain enabled."
        fi
    else
        if [ "$log_mode" = "log" ]; then
            log_info "Building CPU-only (use --with-gpu to enable GPU acceleration)"
        fi
    fi
}

# Build NeurondB
build_neurondb() {
    log_info "Building NeurondB..."
    
    # Detect pg_config path
    if command -v pg_config >/dev/null 2>&1; then
        PG_CONFIG=$(command -v pg_config)
    elif [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
        PG_CONFIG="/usr/lib/postgresql/$PG_VERSION/bin/pg_config"
    elif [ "$OS" = "rocky" ] || [ "$OS" = "rhel" ] || [ "$OS" = "centos" ]; then
        PG_CONFIG="/usr/pgsql-$PG_VERSION/bin/pg_config"
    elif [ "$OS" = "macos" ]; then
        PG_CONFIG="/opt/homebrew/opt/postgresql@$PG_VERSION/bin/pg_config"
    else
        log_error "Cannot find pg_config"
        exit 1
    fi
    
    log_info "Using pg_config: $PG_CONFIG"
    
    # Detect pg_config and PostgreSQL version
    PG_CONFIG_BIN=${PG_CONFIG:-$(command -v pg_config || true)}
    if [ -z "$PG_CONFIG_BIN" ]; then
        echo "[ERROR] pg_config not found. Install PostgreSQL 16/17/18 first." >&2
        exit 1
    fi

    PG_VERSION_STR=$($PG_CONFIG_BIN --version 2>/dev/null || echo "")
    PG_MAJOR=$(echo "$PG_VERSION_STR" | sed -n 's/.* \([0-9][0-9]*\)\..*/\1/p')
    if [ -z "$PG_MAJOR" ]; then
        echo "[ERROR] Unable to parse PostgreSQL version from: $PG_VERSION_STR" >&2
        exit 1
    fi

    case "$PG_MAJOR" in
        16|17|18)
            : # supported
            ;;
        *)
            echo "[ERROR] Unsupported PostgreSQL major version: $PG_MAJOR" >&2
            echo "        NeurondB supports only PostgreSQL 16, 17, and 18." >&2
            exit 1
            ;;
    esac
    
    prepare_make_args
    
    # Generate config.h based on detected GPU support
    log_info "Generating config.h..."
    cat > include/neurondb_config.h <<EOF
/*-------------------------------------------------------------------------
 *
 * neurondb_config.h
 *     Auto-generated configuration header for NeurondB
 *
 * This file is generated by build.sh and should not be edited manually.
 * It defines compile-time macros for GPU support detection.
 *
 * Generated: $(date)
 * Platform: $OS $VERSION
 * PostgreSQL: $PG_VERSION
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_CONFIG_H
#define NEURONDB_CONFIG_H

/* Platform detection */
#define NEURONDB_VERSION "1.0"
#define NEURONDB_BUILD_DATE "$(date +%Y%m%d)"

/* GPU support flags */
EOF

    if [ "$WITH_GPU" = "yes" ] && [ -n "$CUDA_PATH" ]; then
        cat >> include/neurondb_config.h <<EOF
#define NDB_GPU_CUDA 1
#define CUDA_PATH "$CUDA_PATH"
EOF
        log_success "CUDA support enabled in config.h"
    else
        cat >> include/neurondb_config.h <<EOF
/* #undef NDB_GPU_CUDA */
EOF
    fi

    if [ "$WITH_GPU" = "yes" ] && [ -n "$ROCM_PATH" ]; then
        cat >> include/neurondb_config.h <<EOF
#define NDB_GPU_HIP 1
#define ROCM_PATH "$ROCM_PATH"
EOF
        log_success "ROCm support enabled in config.h"
    else
        cat >> include/neurondb_config.h <<EOF
/* #undef NDB_GPU_HIP */
EOF
    fi

    if [ "$WITH_GPU" = "yes" ] && [ -n "$ONNX_PATH" ]; then
        cat >> include/neurondb_config.h <<EOF
#define HAVE_ONNXRUNTIME_GPU 1
#define ONNX_PATH "$ONNX_PATH"
EOF
        log_success "ONNX Runtime GPU enabled in config.h"
    else
        cat >> include/neurondb_config.h <<EOF
/* #undef HAVE_ONNXRUNTIME_GPU */
EOF
    fi

    cat >> include/neurondb_config.h <<EOF

/* PostgreSQL version */
#define NEURONDB_PG_VERSION $PG_VERSION

/* Build configuration */
#define NEURONDB_CPU_FALLBACK 1

#endif /* NEURONDB_CONFIG_H */
EOF

    log_success "config.h generated successfully"
    cat include/neurondb_config.h
    echo ""
    
    # Clean previous build
    make clean PG_CONFIG=$PG_CONFIG $MAKE_ARGS 2>/dev/null || true
    
    # Build
    log_info "Compiling NeurondB..."
    if make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) PG_CONFIG=$PG_CONFIG $MAKE_ARGS all; then
        log_success "Build completed successfully"
    else
        log_error "Build failed"
        exit 1
    fi
    
    # Show build summary
    log_info "Build summary:"
    if [ -f neurondb.so ] || [ -f neurondb.dylib ]; then
        ls -lh neurondb.* | grep -v ".o$"
        
        # Check for GPU libraries linked
        if command -v ldd >/dev/null 2>&1; then
            if ldd neurondb.so 2>/dev/null | grep -q "cuda\|cublas"; then
                log_success "GPU support: CUDA linked"
            elif ldd neurondb.so 2>/dev/null | grep -q "hip\|rocblas"; then
                log_success "GPU support: ROCm linked"
            else
                log_info "GPU support: CPU-only build"
            fi
        elif command -v otool >/dev/null 2>&1; then
            if otool -L neurondb.dylib 2>/dev/null | grep -q "cuda\|cublas"; then
                log_success "GPU support: CUDA linked"
            else
                log_info "GPU support: CPU-only build"
            fi
        fi
    else
        log_error "Shared library not found"
        exit 1
    fi
}

# Install NeurondB
install_neurondb() {
    log_info "Installing NeurondB..."
    
    SUDO=""
    if [ "$(id -u)" != "0" ] && [ "$OS" != "macos" ]; then
        SUDO="sudo"
    fi
    
    prepare_make_args quiet
    if $SUDO make install PG_CONFIG=$PG_CONFIG $MAKE_ARGS; then
        log_success "NeurondB installed successfully"
        
        # Show installation paths
        SHAREDIR=$($PG_CONFIG --sharedir)
        PKGLIBDIR=$($PG_CONFIG --pkglibdir)
        
        log_info "Installation paths:"
        echo "  Extension SQL: $SHAREDIR/extension/neurondb--1.0.sql"
        echo "  Control file:  $SHAREDIR/extension/neurondb.control"
        echo "  Library:       $PKGLIBDIR/neurondb.so (or .dylib)"
    else
        log_error "Installation failed"
        exit 1
    fi
}

# Run regression tests
run_tests() {
    log_info "Running regression tests..."
    
    # Check if PostgreSQL is running
    if ! $PG_CONFIG --bindir/psql -c "SELECT 1" >/dev/null 2>&1; then
        log_warning "PostgreSQL server not running, skipping tests"
        log_info "To run tests later: make installcheck PG_CONFIG=$PG_CONFIG"
        return
    fi
    
    prepare_make_args quiet
    if make installcheck PG_CONFIG=$PG_CONFIG $MAKE_ARGS; then
        log_success "All regression tests passed"
    else
        log_error "Regression tests failed"
        if [ -f regression.diffs ]; then
            log_error "See regression.diffs for details"
            cat regression.diffs
        fi
        exit 1
    fi
}

# Print usage
usage() {
    cat <<EOF
NeurondB Build Script

Usage: $0 [OPTIONS]

Options:
    --install-deps       Install platform dependencies (default: yes)
    --build              Build NeurondB (default: yes)
    --install            Install NeurondB (default: yes)
    --test               Run regression tests (default: no)
    --pg-version VERSION PostgreSQL version (default: auto-detect or 17)
    --with-gpu           Enable GPU support (CUDA/ROCm) (default: no)
    --cuda-path PATH     Path to CUDA toolkit (auto-detect if not specified)
    --rocm-path PATH     Path to ROCm (auto-detect if not specified)
    --onnx-path PATH     Path to ONNX Runtime (auto-detect if not specified)
    --clean              Clean build artifacts only
    --help               Show this help message

Examples:
    $0                                    # Install deps, build, and install (CPU-only)
    $0 --test                             # Full build with tests
    $0 --pg-version 16                    # Build for PostgreSQL 16
    $0 --with-gpu                         # Build with GPU support (auto-detect paths)
    $0 --with-gpu --cuda-path /opt/cuda   # Build with GPU at custom path
    $0 --clean                            # Clean only
    $0 --build --install --test           # Build, install, and test

Environment Variables:
    PG_CONFIG            Path to pg_config (auto-detected if not set)

EOF
    exit 0
}

# Parse command line arguments
INSTALL_DEPS="yes"
BUILD="yes"
INSTALL="yes"
RUN_TESTS="no"
WITH_GPU="${WITH_GPU:-auto}"
CUDA_PATH="${CUDA_PATH:-${CUDA_HOME:-}}"
ROCM_PATH="${ROCM_PATH:-${ROCM_HOME:-}}"
ONNX_PATH="${ONNX_PATH:-}"
ONNX_RUNTIME_PATH="${ONNX_RUNTIME_PATH:-}"
CLEAN_ONLY="no"

while [ $# -gt 0 ]; do
    case "$1" in
        --install-deps)
            INSTALL_DEPS="yes"
            shift
            ;;
        --no-deps)
            INSTALL_DEPS="no"
            shift
            ;;
        --build)
            BUILD="yes"
            shift
            ;;
        --install)
            INSTALL="yes"
            shift
            ;;
        --test)
            RUN_TESTS="yes"
            shift
            ;;
        --pg-version)
            PG_VERSION="$2"
            shift 2
            ;;
        --with-gpu)
            WITH_GPU="yes"
            shift
            ;;
        --cuda-path)
            CUDA_PATH="$2"
            shift 2
            ;;
        --rocm-path)
            ROCM_PATH="$2"
            shift 2
            ;;
        --onnx-path)
            ONNX_PATH="$2"
            ONNX_RUNTIME_PATH="$2"
            shift 2
            ;;
        --clean)
            CLEAN_ONLY="yes"
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Clean only mode
if [ "$CLEAN_ONLY" = "yes" ]; then
    log_info "Cleaning build artifacts..."
    make clean 2>/dev/null || true
    rm -rf pgdata/ regression.diffs regression.out results/ tmp_check/ log/
    log_success "Clean completed"
    exit 0
fi

# Main execution
main() {
    echo "=================================================="
    echo "  NeurondB Build Script"
    echo "  Copyright (c) 2024-2025, pgElephant, Inc."
    echo "=================================================="
    echo ""
    
    # Detect platform and PostgreSQL
    detect_platform
    detect_postgres_version

    # Auto-detect ONNX Runtime if not explicitly provided
    if [ -z "$ONNX_PATH" ]; then
        for p in "$ONNX_RUNTIME_PATH" /usr/local/onnxruntime /usr/lib/onnxruntime /opt/onnxruntime "$HOME/onnxruntime-gpu"; do
            if [ -n "$p" ] && [ -f "$p/include/onnxruntime/core/session/onnxruntime_cxx_api.h" ]; then
                ONNX_PATH="$p"
                break
            fi
        done
    fi
    if [ -z "$ONNX_RUNTIME_PATH" ] && [ -n "$ONNX_PATH" ]; then
        ONNX_RUNTIME_PATH="$ONNX_PATH"
    fi
    
    # Install dependencies
    if [ "$INSTALL_DEPS" = "yes" ]; then
        case "$OS" in
            ubuntu|debian)
                install_ubuntu_deps
                ;;
            rocky|rhel|centos|almalinux)
                install_rocky_deps
                ;;
            macos)
                install_macos_deps
                ;;
            *)
                log_error "Unsupported platform: $OS"
                exit 1
                ;;
        esac
        echo ""
    fi
    
    # Re-detect pg_config after installation
    if [ "$INSTALL_DEPS" = "yes" ]; then
        detect_postgres_version
    fi
    
    detect_onnx_runtime
    detect_cuda_toolkit
    detect_rocm_toolkit

    if [ "$WITH_GPU" = "auto" ]; then
        if [ -n "$CUDA_PATH" ] || [ -n "$ROCM_PATH" ]; then
            WITH_GPU="yes"
            log_info "Auto-detected GPU toolkits. Enabling GPU build."
        else
            WITH_GPU="no"
            log_info "No GPU toolkit detected. Building CPU-only."
        fi
    fi
    
    # Build
    if [ "$BUILD" = "yes" ]; then
        build_neurondb
        echo ""
    fi
    
    # Install
    if [ "$INSTALL" = "yes" ]; then
        install_neurondb
        echo ""
    fi
    
    # Test
    if [ "$RUN_TESTS" = "yes" ]; then
        run_tests
        echo ""
    fi
    
    # Final summary
    echo "=================================================="
    log_success "NeurondB build process completed"
    echo "=================================================="
    echo ""
    echo "Next steps:"
    echo "  1. Add to postgresql.conf:"
    echo "     shared_preload_libraries = 'neurondb'"
    echo ""
    echo "  2. Restart PostgreSQL"
    echo ""
    echo "  3. Create extension:"
    echo "     psql -d mydb -c 'CREATE EXTENSION neurondb;'"
    echo ""
    echo "  4. Optional GPU configuration:"
    echo "     SET neurondb.gpu_enabled = on;"
    echo "     SET neurondb.gpu_device = 0;"
    echo ""
    echo "For documentation, visit:"
    echo "  https://github.com/pgElephant/NeurondB"
    echo ""
}

# Run main
main

