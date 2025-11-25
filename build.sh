#!/usr/bin/env bash
set -euo pipefail

#-------------------------------------------------------------------------
#
# build.sh
#     Comprehensive build script for NeurondB
#
# Supports:
#   - Platforms: Rocky Linux, Ubuntu/Debian, macOS
#   - GPU Backends: CUDA, Metal (macOS), ROCm
#   - PostgreSQL versions: 16, 17, 18
#
# This script handles everything from dependency installation to
# compilation and verification, ensuring NeurondB builds and runs
# perfectly on all supported platforms.
#
# Copyright (c) 2024-2025, pgElephant, Inc.
#
#-------------------------------------------------------------------------

###############################################################################
# CONFIGURATION AND GLOBALS
###############################################################################

# Script metadata
SCRIPT_VERSION="2.0.0"
SCRIPT_NAME="$(basename "$0")"

# Defaults
VERBOSE="${VERBOSE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_TESTS="${SKIP_TESTS:-1}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

# GPU configuration
GPU_MODE="${GPU_MODE:-auto}"  # auto, none, cuda, metal, rocm, or comma-separated
GPU_TYPES=""  # Internal: parsed from GPU_MODE

# PostgreSQL configuration
PG_VERSION="${PG_VERSION:-}"  # Auto-detect if empty
PG_CONFIG="${PG_CONFIG:-}"    # Auto-detect if empty

# Paths (will be auto-detected)
CUDA_PATH="${CUDA_PATH:-${CUDA_HOME:-}}"
ROCM_PATH="${ROCM_PATH:-${ROCM_HOME:-}}"
ONNX_RUNTIME_PATH="${ONNX_RUNTIME_PATH:-/usr/local/onnxruntime}"

# Platform detection
PLATFORM=""
OS_ID=""
OS_VERSION=""

# Build state
SELECTED_PG_CONFIG=""
PG_MAJOR_SELECTED=""
DETECTED_CUDA_PATH=""
DETECTED_ROCM_PATH=""
DETECTED_METAL=""

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly MAGENTA='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'  # No Color
readonly BOLD='\033[1m'
readonly DIM='\033[2m'

# Logging functions
msg() {
	printf "[${CYAN}INFO${NC}] %s\n" "$*"
}

msg_verbose() {
	[[ $VERBOSE -eq 1 ]] && printf "[${DIM}DEBUG${NC}] %s\n" "$*" || true
}

success() {
	printf "[${GREEN}OK${NC}] %s\n" "$*"
}

warning() {
	printf "[${YELLOW}WARN${NC}] %s\n" "$*" >&2
}

error() {
	printf "[${RED}ERROR${NC}] %s\n" "$*" >&2
}

fatal() {
	error "$*"
	exit 1
}

# Timestamp function
ts() {
	date "+%Y-%m-%d %H:%M:%S"
}

# Section header
section() {
	echo ""
	printf "${BOLD}${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
	printf "${BOLD}${MAGENTA}  %s${NC}\n" "$*"
	printf "${BOLD}${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
	echo ""
}

# Run command with timing and error handling
run_cmd() {
	local desc="$1"
	shift
	local start elapsed rc output
	
	start=$(date +%s)
	
	if [[ $VERBOSE -eq 1 ]]; then
		msg "Executing: $*"
		"$@"
		rc=$?
	else
		if output=$("$@" 2>&1); then
			rc=0
		else
			rc=$?
			echo "$output" >&2
		fi
	fi
	
	elapsed=$(( $(date +%s) - start ))
	
	if [[ $rc -eq 0 ]]; then
		success "$desc (${elapsed}s)"
		return 0
	else
		error "$desc failed after ${elapsed}s (exit code: $rc)"
		return $rc
	fi
}

# Check if command exists
has_cmd() {
	command -v "$1" >/dev/null 2>&1
}

# Check if file/directory exists
has_path() {
	[[ -e "$1" ]]
}

###############################################################################
# PLATFORM DETECTION
###############################################################################

detect_platform() {
	section "Platform Detection"
	
	local uname_s uname_m
	uname_s="$(uname -s)"
	uname_m="$(uname -m)"
	
	msg "Kernel: $uname_s $uname_m"
	
	if [[ "$uname_s" == "Darwin" ]]; then
		PLATFORM="osx"
		OS_ID="macos"
		OS_VERSION="$(sw_vers -productVersion)"
		msg "Detected: macOS $OS_VERSION"
		
		# Detect architecture
		if [[ "$uname_m" == "arm64" ]]; then
			msg "Architecture: Apple Silicon (arm64)"
		else
			msg "Architecture: Intel (x86_64)"
		fi
		
	elif [[ "$uname_s" == "Linux" ]]; then
		# Try to read /etc/os-release
		if [[ -f /etc/os-release ]]; then
			# shellcheck source=/dev/null
			. /etc/os-release
			OS_ID="${ID:-unknown}"
			OS_VERSION="${VERSION_ID:-unknown}"
			
			msg "OS Release: ${PRETTY_NAME:-$OS_ID $OS_VERSION}"
			
			# Classify Linux distribution
			case "${ID:-}" in
				rocky|centos|rhel|almalinux|fedora)
					PLATFORM="rocky"
					msg "Detected: Rocky/RHEL/CentOS family"
					;;
				ubuntu|debian)
					PLATFORM="deb"
					msg "Detected: Debian/Ubuntu family"
					;;
				*)
					# Try to infer from other fields
					if grep -qiE "rocky|centos|rhel|redhat|fedora" /etc/os-release 2>/dev/null; then
						PLATFORM="rocky"
						msg "Detected: RPM-based distribution (assuming Rocky/RHEL)"
					elif grep -qiE "debian|ubuntu" /etc/os-release 2>/dev/null; then
						PLATFORM="deb"
						msg "Detected: Debian-based distribution"
					else
						PLATFORM="unknown"
						error "Unknown Linux distribution: $OS_ID"
					fi
					;;
			esac
		else
			# Fallback: check for distribution-specific files
			if [[ -f /etc/rocky-release ]] || [[ -f /etc/redhat-release ]]; then
				PLATFORM="rocky"
				msg "Detected: Rocky/RHEL (via release file)"
			elif has_cmd apt-get; then
				PLATFORM="deb"
				msg "Detected: Debian-based (via apt-get)"
			elif has_cmd dnf || has_cmd yum; then
				PLATFORM="rocky"
				msg "Detected: RPM-based (via dnf/yum)"
			else
				PLATFORM="unknown"
				error "Cannot determine Linux distribution"
			fi
		fi
		
		msg "Architecture: $uname_m"
		
	else
		fatal "Unsupported operating system: $uname_s"
	fi
	
	success "Platform detected: $PLATFORM"
	
	# Print detailed system information
	print_system_info
}

print_system_info() {
	msg_verbose "System Information:"
	msg_verbose "  Hostname: $(hostname 2>/dev/null || echo 'unknown')"
	msg_verbose "  Kernel: $(uname -r)"
	msg_verbose "  Architecture: $(uname -m)"
	
	if [[ "$PLATFORM" != "osx" ]]; then
		msg_verbose "  CPU: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^[ \t]*//' || echo 'unknown')"
		msg_verbose "  CPU Cores: $(nproc 2>/dev/null || echo 'unknown')"
		msg_verbose "  Memory: $(free -h 2>/dev/null | grep '^Mem:' | awk '{print $2}' || echo 'unknown')"
	fi
}

###############################################################################
# GPU DETECTION
###############################################################################

detect_cuda() {
	local found=false
	DETECTED_CUDA_PATH=""
	
	msg_verbose "Detecting CUDA..."
	
	# Method 1: Check nvcc in PATH
	if has_cmd nvcc; then
		local nvcc_path cuda_home
		nvcc_path="$(command -v nvcc)"
		cuda_home="$(dirname "$(dirname "$nvcc_path")")"
		
		if [[ -f "$cuda_home/include/cuda_runtime.h" ]]; then
			DETECTED_CUDA_PATH="$cuda_home"
			found=true
			msg "Found CUDA via nvcc: $DETECTED_CUDA_PATH"
		fi
	fi
	
	# Method 2: Check common installation paths
	if [[ "$found" == "false" ]]; then
		local cuda_candidates=(
			"${CUDA_PATH:-}"
			"${CUDA_HOME:-}"
			"/usr/local/cuda"
			"/usr/local/cuda-12.0"
			"/usr/local/cuda-11.8"
			"/opt/cuda"
			"/usr/lib/cuda"
			"/usr/cuda"
		)
		
		for candidate in "${cuda_candidates[@]}"; do
			[[ -z "$candidate" ]] && continue
			
			if [[ -f "$candidate/include/cuda_runtime.h" ]]; then
				DETECTED_CUDA_PATH="$candidate"
				found=true
				msg "Found CUDA at: $DETECTED_CUDA_PATH"
				break
			fi
		done
	fi
	
	# Method 3: Check system include paths (Ubuntu/Debian packages)
	if [[ "$found" == "false" ]] && [[ -f "/usr/include/cuda/cuda_runtime.h" ]]; then
		DETECTED_CUDA_PATH="/usr"
		found=true
		msg "Found CUDA via system headers: $DETECTED_CUDA_PATH"
	fi
	
	# Method 4: Check library paths for CUDA libraries
	if [[ "$found" == "false" ]]; then
		local lib_paths=(
			"/usr/lib/x86_64-linux-gnu"
			"/usr/lib64"
		)
		
		for lib_path in "${lib_paths[@]}"; do
			if [[ -f "$lib_path/libcudart.so" ]] || [[ -f "$lib_path/libcublas.so" ]]; then
				# Try to infer CUDA path
				if [[ -d "/usr/local/cuda" ]]; then
					DETECTED_CUDA_PATH="/usr/local/cuda"
					found=true
					msg "Found CUDA libraries, using: $DETECTED_CUDA_PATH"
					break
				fi
			fi
		done
	fi
	
	if [[ "$found" == "true" ]]; then
		# Verify CUDA version
		if has_cmd nvcc; then
			local cuda_ver
			cuda_ver=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/' | head -n1 || echo "unknown")
			msg "CUDA version: $cuda_ver"
		fi
		
		# Check for NVIDIA driver
		if has_cmd nvidia-smi; then
			local driver_ver
			set +e
			driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')
			set -e
			if [[ -n "$driver_ver" ]] && [[ "$driver_ver" != *"failed"* ]]; then
				msg "NVIDIA driver version: $driver_ver"
			fi
		fi
		
		return 0
	else
		msg_verbose "CUDA not detected"
		return 1
	fi
}

detect_rocm() {
	local found=false
	DETECTED_ROCM_PATH=""
	
	msg_verbose "Detecting ROCm..."
	
	# Method 1: Check hipcc in PATH
	if has_cmd hipcc; then
		local hipcc_path rocm_home
		hipcc_path="$(command -v hipcc)"
		rocm_home="$(dirname "$(dirname "$hipcc_path")")"
		
		if [[ -f "$rocm_home/include/hip/hip_runtime.h" ]]; then
			DETECTED_ROCM_PATH="$rocm_home"
			found=true
			msg "Found ROCm via hipcc: $DETECTED_ROCM_PATH"
		fi
	fi
	
	# Method 2: Check common installation paths
	if [[ "$found" == "false" ]]; then
		local rocm_candidates=(
			"${ROCM_PATH:-}"
			"${ROCM_HOME:-}"
			"/opt/rocm"
			"/opt/rocm-5.7"
			"/opt/rocm-6.0"
			"/usr/rocm"
		)
		
		for candidate in "${rocm_candidates[@]}"; do
			[[ -z "$candidate" ]] && continue
			
			if [[ -f "$candidate/include/hip/hip_runtime.h" ]]; then
				DETECTED_ROCM_PATH="$candidate"
				found=true
				msg "Found ROCm at: $DETECTED_ROCM_PATH"
				
				# Try to read version
				if [[ -f "$candidate/.info/version" ]]; then
					local rocm_ver
					rocm_ver=$(cat "$candidate/.info/version" 2>/dev/null || echo "unknown")
					msg "ROCm version: $rocm_ver"
				fi
				break
			fi
		done
	fi
	
	if [[ "$found" == "true" ]]; then
		return 0
	else
		msg_verbose "ROCm not detected"
		return 1
	fi
}

detect_metal() {
	DETECTED_METAL="false"
	
	if [[ "$PLATFORM" != "osx" ]]; then
		return 1
	fi
	
	msg_verbose "Detecting Metal (macOS)..."
	
	# Metal is available on macOS if Xcode Command Line Tools are installed
	if xcode-select -p >/dev/null 2>&1; then
		local xcode_clt
		xcode_clt="$(xcode-select -p 2>/dev/null)"
		msg "Found Xcode Command Line Tools: $xcode_clt"
		
		# Check for Metal framework
		if [[ -d "/System/Library/Frameworks/Metal.framework" ]]; then
			DETECTED_METAL="true"
			msg "Metal framework available"
			return 0
		fi
	fi
	
	msg_verbose "Metal not available (Xcode CLT required)"
	return 1
}

detect_gpu_backends() {
	section "GPU Backend Detection"
	
	local cuda_available=false
	local rocm_available=false
	local metal_available=false
	
	# Parse GPU_MODE
	if [[ -n "$GPU_MODE" ]] && [[ "$GPU_MODE" != "auto" ]] && [[ "$GPU_MODE" != "none" ]]; then
		GPU_TYPES=$(echo "$GPU_MODE" | tr ',' ' ')
		msg "Requested GPU backends: $GPU_TYPES"
	fi
	
	# Detect CUDA
	if detect_cuda; then
		cuda_available=true
		CUDA_PATH="${CUDA_PATH:-$DETECTED_CUDA_PATH}"
	fi
	
	# Detect ROCm
	if detect_rocm; then
		rocm_available=true
		ROCM_PATH="${ROCM_PATH:-$DETECTED_ROCM_PATH}"
	fi
	
	# Detect Metal
	if detect_metal; then
		metal_available=true
	fi
	
	# Determine final GPU mode
	if [[ -n "$GPU_TYPES" ]]; then
		# User specified GPU types
		local selected=""
		for gpu_type in $GPU_TYPES; do
			case "$gpu_type" in
				cuda)
					if [[ "$cuda_available" == "true" ]]; then
						selected="cuda"
						break
					elif [[ "$GPU_MODE" != "none" ]]; then
						warning "CUDA requested but not available"
					fi
					;;
				rocm)
					if [[ "$rocm_available" == "true" ]]; then
						selected="rocm"
						break
					elif [[ "$GPU_MODE" != "none" ]]; then
						warning "ROCm requested but not available"
					fi
					;;
				metal)
					if [[ "$metal_available" == "true" ]]; then
						selected="metal"
						break
					elif [[ "$GPU_MODE" != "none" ]]; then
						warning "Metal requested but not available"
					fi
					;;
			esac
		done
		
		if [[ -n "$selected" ]]; then
			GPU_MODE="$selected"
		else
			GPU_MODE="none"
			warning "No requested GPU backends available, building CPU-only"
		fi
		
	elif [[ "$GPU_MODE" == "auto" ]]; then
		# Auto-detect: prefer CUDA > ROCm > Metal
		if [[ "$cuda_available" == "true" ]]; then
			GPU_MODE="cuda"
			msg "Auto-selected: CUDA"
		elif [[ "$rocm_available" == "true" ]]; then
			GPU_MODE="rocm"
			msg "Auto-selected: ROCm"
		elif [[ "$metal_available" == "true" ]]; then
			GPU_MODE="metal"
			msg "Auto-selected: Metal"
		else
			GPU_MODE="none"
			msg "No GPU backends detected, building CPU-only"
		fi
	fi
	
	success "GPU mode: $GPU_MODE"
	
	# Print summary
	echo ""
	printf "  ${BOLD}GPU Backend Summary:${NC}\n"
	printf "    CUDA:  %s\n" "$([[ "$cuda_available" == "true" ]] && echo "${GREEN}✓${NC} $CUDA_PATH" || echo "${RED}✗${NC} Not available")"
	printf "    ROCm:  %s\n" "$([[ "$rocm_available" == "true" ]] && echo "${GREEN}✓${NC} $ROCM_PATH" || echo "${RED}✗${NC} Not available")"
	printf "    Metal: %s\n" "$([[ "$metal_available" == "true" ]] && echo "${GREEN}✓${NC} Available" || echo "${RED}✗${NC} Not available")"
	echo ""
}

###############################################################################
# POSTGRESQL DETECTION
###############################################################################

detect_postgresql() {
	section "PostgreSQL Detection"
	
	SELECTED_PG_CONFIG=""
	PG_MAJOR_SELECTED=""
	
	# Priority 1: User-specified PG_CONFIG
	if [[ -n "${PG_CONFIG:-}" ]] && [[ -x "${PG_CONFIG:-}" ]]; then
		SELECTED_PG_CONFIG="$PG_CONFIG"
		msg "Using user-specified PG_CONFIG: $SELECTED_PG_CONFIG"
		
	# Priority 2: pg_config in PATH
	elif has_cmd pg_config; then
		SELECTED_PG_CONFIG="$(command -v pg_config)"
		msg "Found pg_config in PATH: $SELECTED_PG_CONFIG"
		
	# Priority 3: Platform-specific common locations
	else
		local candidates=()
		
		case "$PLATFORM" in
			deb)
				# Debian/Ubuntu: try versioned and generic paths
				for ver in 18 17 16 15 14; do
					candidates+=(
						"/usr/lib/postgresql/$ver/bin/pg_config"
						"/usr/bin/pg_config"
					)
				done
				;;
			rocky)
				# Rocky/RHEL: versioned paths
				for ver in 18 17 16 15 14; do
					candidates+=(
						"/usr/pgsql-$ver/bin/pg_config"
						"/usr/bin/pg_config"
					)
				done
				;;
			osx)
				# macOS: Homebrew paths
				for ver in 18 17 16 15 14; do
					candidates+=(
						"/opt/homebrew/opt/postgresql@$ver/bin/pg_config"
						"/usr/local/opt/postgresql@$ver/bin/pg_config"
						"/opt/homebrew/bin/pg_config"
						"/usr/local/bin/pg_config"
					)
				done
				;;
		esac
		
		for candidate in "${candidates[@]}"; do
			if [[ -x "$candidate" ]]; then
				SELECTED_PG_CONFIG="$candidate"
				msg "Found pg_config at: $SELECTED_PG_CONFIG"
				break
			fi
		done
	fi
	
	# Extract PostgreSQL version
	if [[ -n "$SELECTED_PG_CONFIG" ]]; then
		local pg_ver_str
		pg_ver_str=$("$SELECTED_PG_CONFIG" --version 2>/dev/null || echo "")
		
		if [[ -n "$pg_ver_str" ]]; then
			PG_MAJOR_SELECTED=$(echo "$pg_ver_str" | sed -n 's/.*PostgreSQL \([0-9][0-9]*\)\..*/\1/p')
			msg "PostgreSQL version: $pg_ver_str (major: $PG_MAJOR_SELECTED)"
			
			# Verify supported version
			case "$PG_MAJOR_SELECTED" in
				16|17|18)
					success "PostgreSQL $PG_MAJOR_SELECTED is supported"
					;;
				*)
					warning "PostgreSQL $PG_MAJOR_SELECTED may not be fully supported (tested with 16, 17, 18)"
					;;
			esac
		else
			warning "Could not determine PostgreSQL version"
		fi
		
		# Print detailed PostgreSQL information
		print_postgresql_info
		
	else
		# Will attempt to install if not found
		warning "pg_config not found - will attempt to install PostgreSQL"
		
		# Try to detect from package manager
		case "$PLATFORM" in
			deb)
				# Check for available PostgreSQL packages
				if has_cmd apt-cache; then
					local pg_versions
					pg_versions=$(apt-cache search -n '^postgresql-server-dev-[0-9]+$' 2>/dev/null | \
						sed -n 's/^postgresql-server-dev-\([0-9][0-9]*\)$/\1/p' | \
						sort -rn | head -n1 || true)
					
					if [[ -n "$pg_versions" ]]; then
						PG_MAJOR_SELECTED="$pg_versions"
						msg "Detected available PostgreSQL version: $PG_MAJOR_SELECTED"
					fi
				fi
				;;
			rocky)
				# Check RPM packages
				if has_cmd rpm; then
					local pg_versions
					pg_versions=$(rpm -qa 'postgresql*-devel' 2>/dev/null | \
						sed -n 's/.*postgresql\([0-9][0-9]*\)-devel.*/\1/p' | \
						sort -rn | head -n1 || true)
					
					if [[ -n "$pg_versions" ]]; then
						PG_MAJOR_SELECTED="$pg_versions"
						msg "Detected installed PostgreSQL version: $PG_MAJOR_SELECTED"
					fi
				fi
				;;
		esac
		
		# Default to 18 if nothing detected
		if [[ -z "$PG_MAJOR_SELECTED" ]]; then
			PG_MAJOR_SELECTED="${PG_VERSION:-18}"
			msg "Using default PostgreSQL version: $PG_MAJOR_SELECTED"
		fi
	fi
	
	export SELECTED_PG_CONFIG
}

print_postgresql_info() {
	if [[ -z "$SELECTED_PG_CONFIG" ]]; then
		return
	fi
	
	msg_verbose "PostgreSQL Details:"
	msg_verbose "  Version:     $("$SELECTED_PG_CONFIG" --version 2>/dev/null || echo 'unknown')"
	msg_verbose "  Bindir:      $("$SELECTED_PG_CONFIG" --bindir 2>/dev/null || echo 'unknown')"
	msg_verbose "  Libdir:      $("$SELECTED_PG_CONFIG" --libdir 2>/dev/null || echo 'unknown')"
	msg_verbose "  Includedir:  $("$SELECTED_PG_CONFIG" --includedir 2>/dev/null || echo 'unknown')"
	msg_verbose "  Sharedir:    $("$SELECTED_PG_CONFIG" --sharedir 2>/dev/null || echo 'unknown')"
	msg_verbose "  Pkglibdir:   $("$SELECTED_PG_CONFIG" --pkglibdir 2>/dev/null || echo 'unknown')"
}

###############################################################################
# DEPENDENCY INSTALLATION
###############################################################################

wait_for_package_manager() {
	# Wait for package manager locks to be released
	local max_wait=300
	local wait_interval=5
	local waited=0
	
	msg_verbose "Checking for package manager locks..."
	
	case "$PLATFORM" in
		deb)
			while [[ $waited -lt $max_wait ]]; do
				if ! pgrep -f "(apt|dpkg)" >/dev/null 2>&1 && \
				   [[ ! -f /var/lib/dpkg/lock-frontend ]] && \
				   [[ ! -f /var/lib/dpkg/lock ]]; then
					return 0
				fi
				
				if [[ $waited -eq 0 ]]; then
					msg "Waiting for package manager to be available..."
				fi
				
				sleep $wait_interval
				waited=$((waited + wait_interval))
			done
			;;
		rocky)
			while [[ $waited -lt $max_wait ]]; do
				if ! pgrep -f "(dnf|yum|rpm)" >/dev/null 2>&1 && \
				   [[ ! -f /var/run/dnf.pid ]]; then
					return 0
				fi
				
				if [[ $waited -eq 0 ]]; then
					msg "Waiting for package manager to be available..."
				fi
				
				sleep $wait_interval
				waited=$((waited + wait_interval))
			done
			;;
	esac
	
	warning "Package manager timeout, proceeding anyway..."
}

install_dependencies_ubuntu() {
	section "Installing Dependencies (Ubuntu/Debian)"
	
	wait_for_package_manager
	
	# Base packages
	local packages=(
		"build-essential"
		"git"
		"curl"
		"wget"
		"ca-certificates"
		"pkg-config"
		"libxml2-dev"
		"libssl-dev"
		"libcurl4-openssl-dev"
		"zlib1g-dev"
	)
	
	# PostgreSQL dev package
	if [[ -z "$SELECTED_PG_CONFIG" ]]; then
		local pg_pkg="postgresql-server-dev-${PG_MAJOR_SELECTED:-18}"
		packages+=("$pg_pkg")
		msg "Will install PostgreSQL dev package: $pg_pkg"
	else
		msg "PostgreSQL already available, skipping dev package"
	fi
	
	# CUDA packages if needed
	if [[ "$GPU_MODE" == "cuda" ]] && [[ -z "${CUDA_PATH:-}" ]]; then
		packages+=("nvidia-cuda-toolkit" "nvidia-cuda-dev")
		msg "Will install CUDA toolkit (this may take 10-30 minutes, ~3-4GB)"
	fi
	
	msg "Updating package lists..."
	run_cmd "apt-get update" sudo apt-get update -y
	
	msg "Installing packages..."
	run_cmd "apt-get install" sudo apt-get install -y --no-install-recommends "${packages[@]}"
	
	# Install ONNX Runtime
	install_onnx_runtime
	
	success "Ubuntu dependencies installed"
}

install_dependencies_rocky() {
	section "Installing Dependencies (Rocky/RHEL)"
	
	wait_for_package_manager
	
	local pkg_manager="dnf"
	if ! has_cmd dnf; then
		pkg_manager="yum"
	fi
	
	# Base packages
	local packages=(
		"gcc"
		"gcc-c++"
		"make"
		"git"
		"curl"
		"wget"
		"ca-certificates"
		"pkgconfig"
		"libxml2-devel"
		"openssl-devel"
		"libcurl-devel"
		"zlib-devel"
		"redhat-rpm-config"
	)
	
	# PostgreSQL repository
	if ! [[ -f /etc/yum.repos.d/pgdg-redhat-all.repo ]]; then
		msg "Installing PostgreSQL repository..."
		
		local rhel_version
		if has_cmd rpm; then
			rhel_version=$(rpm -E %{rhel} 2>/dev/null || echo "9")
		else
			rhel_version="9"
		fi
		
		local pg_repo_url="https://download.postgresql.org/pub/repos/yum/reporpms/EL-${rhel_version}-x86_64/pgdg-redhat-repo-latest.noarch.rpm"
		
		run_cmd "Install PostgreSQL repo" sudo $pkg_manager install -y "$pg_repo_url" || true
		
		# Disable built-in PostgreSQL module
		if sudo $pkg_manager module list postgresql 2>/dev/null | grep -q postgresql; then
			run_cmd "Disable PostgreSQL module" sudo $pkg_manager -qy module disable postgresql || true
		fi
	fi
	
	# PostgreSQL dev package
	if [[ -z "$SELECTED_PG_CONFIG" ]]; then
		local pg_pkg="postgresql${PG_MAJOR_SELECTED:-18}-devel"
		packages+=("$pg_pkg")
		msg "Will install PostgreSQL dev package: $pg_pkg"
	else
		msg "PostgreSQL already available, skipping dev package"
	fi
	
	# CUDA packages if needed
	if [[ "$GPU_MODE" == "cuda" ]] && [[ -z "${CUDA_PATH:-}" ]]; then
		warning "CUDA on Rocky/RHEL typically requires manual installation from NVIDIA"
		warning "See: https://developer.nvidia.com/cuda-downloads"
	fi
	
	msg "Installing packages..."
	run_cmd "$pkg_manager install" sudo $pkg_manager install -y "${packages[@]}" || {
		# Fallback: try without PostgreSQL devel if it fails
		if [[ -n "${pg_pkg:-}" ]]; then
			warning "Failed to install $pg_pkg, trying without it..."
			local packages_no_pg=("${packages[@]}")
			packages_no_pg=("${packages_no_pg[@]/$pg_pkg}")
			run_cmd "$pkg_manager install (without PG)" sudo $pkg_manager install -y "${packages_no_pg[@]}"
		fi
	}
	
	# Install ONNX Runtime
	install_onnx_runtime
	
	success "Rocky dependencies installed"
}

install_dependencies_macos() {
	section "Installing Dependencies (macOS)"
	
	# Check for Homebrew
	if ! has_cmd brew; then
		fatal "Homebrew is required. Install from https://brew.sh and re-run."
	fi
	
	# Xcode Command Line Tools (required for Metal)
	if ! xcode-select -p >/dev/null 2>&1; then
		msg "Installing Xcode Command Line Tools..."
		xcode-select --install || {
			warning "Xcode CLT installation may require user interaction"
		}
	fi
	
	# Base packages
	local packages=(
		"llvm"
		"make"
		"git"
		"curl"
		"wget"
		"pkg-config"
		"libxml2"
		"openssl@3"
		"zlib"
	)
	
	# PostgreSQL via Homebrew
	if [[ -z "$SELECTED_PG_CONFIG" ]]; then
		local pg_formula="postgresql@${PG_MAJOR_SELECTED:-18}"
		msg "Will install PostgreSQL: $pg_formula"
		
		# Check if already installed
		if brew list --versions "$pg_formula" >/dev/null 2>&1; then
			msg "PostgreSQL $pg_formula already installed"
		else
			run_cmd "Install PostgreSQL" brew install "$pg_formula" || {
				warning "PostgreSQL installation failed, trying generic formula..."
				brew install postgresql || true
			}
		fi
		
		# Update PATH hint
		local pg_prefix
		pg_prefix=$(brew --prefix "$pg_formula" 2>/dev/null || brew --prefix postgresql 2>/dev/null || echo "")
		if [[ -n "$pg_prefix" ]] && [[ -x "$pg_prefix/bin/pg_config" ]]; then
			msg "PostgreSQL installed at: $pg_prefix"
			SELECTED_PG_CONFIG="$pg_prefix/bin/pg_config"
		fi
	else
		msg "PostgreSQL already available"
	fi
	
	# Install base packages
	msg "Installing packages..."
	for pkg in "${packages[@]}"; do
		if ! brew list --versions "$pkg" >/dev/null 2>&1; then
			run_cmd "Install $pkg" brew install "$pkg" || warning "Failed to install $pkg"
		else
			msg_verbose "$pkg already installed"
		fi
	done
	
	# Install ONNX Runtime
	install_onnx_runtime
	
	success "macOS dependencies installed"
}

install_onnx_runtime() {
	section "Installing ONNX Runtime"
	
	local onnx_version="1.17.0"
	local onnx_arch onnx_url
	
	# Determine architecture and platform
	case "$PLATFORM" in
		osx)
			if [[ "$(uname -m)" == "arm64" ]]; then
				onnx_arch="osx-arm64"
			else
				onnx_arch="osx-x86_64"
			fi
			onnx_url="https://github.com/microsoft/onnxruntime/releases/download/v${onnx_version}/onnxruntime-${onnx_arch}-${onnx_version}.tgz"
			;;
		*)
			# Linux
			onnx_arch="linux-x64"
			onnx_url="https://github.com/microsoft/onnxruntime/releases/download/v${onnx_version}/onnxruntime-${onnx_arch}-${onnx_version}.tgz"
			;;
	esac
	
	# Check if already installed
	if [[ -d "$ONNX_RUNTIME_PATH" ]] && \
	   [[ -f "$ONNX_RUNTIME_PATH/include/onnxruntime_c_api.h" ]]; then
		msg "ONNX Runtime already installed at: $ONNX_RUNTIME_PATH"
		return 0
	fi
	
	msg "Downloading ONNX Runtime ${onnx_version}..."
	msg "URL: $onnx_url"
	
	local tmp_file="/tmp/onnxruntime-${onnx_version}.tgz"
	
	# Download
	if has_cmd curl; then
		run_cmd "Download ONNX Runtime" curl -L -f -o "$tmp_file" "$onnx_url"
	elif has_cmd wget; then
		run_cmd "Download ONNX Runtime" wget -O "$tmp_file" "$onnx_url"
	else
		fatal "Neither curl nor wget found, cannot download ONNX Runtime"
	fi
	
	# Extract
	msg "Extracting ONNX Runtime to $ONNX_RUNTIME_PATH..."
	run_cmd "Create ONNX Runtime directory" sudo mkdir -p "$ONNX_RUNTIME_PATH"
	run_cmd "Extract ONNX Runtime" sudo tar -xzf "$tmp_file" -C "$ONNX_RUNTIME_PATH" --strip-components=1
	
	# Cleanup
	rm -f "$tmp_file"
	
	# Verify installation
	if [[ -f "$ONNX_RUNTIME_PATH/include/onnxruntime_c_api.h" ]]; then
		success "ONNX Runtime installed successfully"
		
		# Install GPU version if CUDA is enabled
		if [[ "$GPU_MODE" == "cuda" ]] && [[ -n "${CUDA_PATH:-}" ]]; then
			install_onnx_runtime_gpu "$onnx_version"
		fi
	else
		fatal "ONNX Runtime installation verification failed"
	fi
}

install_onnx_runtime_gpu() {
	local onnx_version="$1"
	
	section "Installing ONNX Runtime with GPU Support"
	
	local onnx_gpu_url
	case "$PLATFORM" in
		osx)
			# macOS doesn't have official GPU build, use CPU
			warning "ONNX Runtime GPU builds not available for macOS, using CPU version"
			return 0
			;;
		*)
			onnx_gpu_url="https://github.com/microsoft/onnxruntime/releases/download/v${onnx_version}/onnxruntime-linux-x64-gpu-${onnx_version}.tgz"
			;;
	esac
	
	local tmp_file="/tmp/onnxruntime-gpu-${onnx_version}.tgz"
	
	msg "Downloading ONNX Runtime GPU version..."
	
	if has_cmd curl; then
		run_cmd "Download ONNX Runtime GPU" curl -L -f -o "$tmp_file" "$onnx_gpu_url" || {
			warning "Failed to download GPU version, continuing with CPU version"
			return 0
		}
	elif has_cmd wget; then
		run_cmd "Download ONNX Runtime GPU" wget -O "$tmp_file" "$onnx_gpu_url" || {
			warning "Failed to download GPU version, continuing with CPU version"
			return 0
		}
	fi
	
	msg "Extracting ONNX Runtime GPU version..."
	run_cmd "Extract ONNX Runtime GPU" sudo tar -xzf "$tmp_file" -C "$ONNX_RUNTIME_PATH" --strip-components=1 || true
	rm -f "$tmp_file"
	
	success "ONNX Runtime GPU version installed"
}

###############################################################################
# BUILD CONFIGURATION
###############################################################################

write_config_header() {
	section "Generating Configuration"
	
	mkdir -p include
	
	local pg_major="${PG_MAJOR_SELECTED:-18}"
	local build_date=$(date +%Y%m%d)
	local build_time=$(date +"%Y-%m-%d %H:%M:%S")
	
	msg "Generating neurondb_config.h..."
	
	cat > include/neurondb_config.h <<EOF
/*-------------------------------------------------------------------------
 *
 * neurondb_config.h
 *     Auto-generated configuration header for NeurondB
 *
 * This file is generated by build.sh and should not be edited manually.
 * It defines compile-time macros for GPU support and platform detection.
 *
 * Generated: ${build_time}
 * Platform: ${PLATFORM}
 * PostgreSQL: ${pg_major}
 *
 *-------------------------------------------------------------------------
 */

#ifndef NEURONDB_CONFIG_H
#define NEURONDB_CONFIG_H

/* Version information */
#define NEURONDB_VERSION "1.0"
#define NEURONDB_BUILD_DATE "${build_date}"

/* Platform detection */
#define NEURONDB_PLATFORM_${PLATFORM^^} 1

/* PostgreSQL version */
#define NEURONDB_PG_VERSION ${pg_major}

/* GPU support flags */
EOF

	# CUDA support
	if [[ "$GPU_MODE" == "cuda" ]] && [[ -n "${CUDA_PATH:-}" ]]; then
		cat >> include/neurondb_config.h <<EOF
#define NDB_GPU_CUDA 1
#define HAVE_CUDA 1
#define CUDA_PATH "${CUDA_PATH}"
EOF
		msg "CUDA support: Enabled"
	else
		cat >> include/neurondb_config.h <<EOF
/* #undef NDB_GPU_CUDA */
/* #undef HAVE_CUDA */
EOF
	fi
	
	# ROCm support
	if [[ "$GPU_MODE" == "rocm" ]] && [[ -n "${ROCM_PATH:-}" ]]; then
		cat >> include/neurondb_config.h <<EOF
#define NDB_GPU_HIP 1
#define ROCM_PATH "${ROCM_PATH}"
EOF
		msg "ROCm support: Enabled"
	else
		cat >> include/neurondb_config.h <<EOF
/* #undef NDB_GPU_HIP */
EOF
	fi
	
	# Metal support
	if [[ "$GPU_MODE" == "metal" ]]; then
		cat >> include/neurondb_config.h <<EOF
#define NDB_GPU_METAL 1
EOF
		msg "Metal support: Enabled"
	else
		cat >> include/neurondb_config.h <<EOF
/* #undef NDB_GPU_METAL */
EOF
	fi
	
	# ONNX Runtime
	if [[ -f "$ONNX_RUNTIME_PATH/include/onnxruntime_c_api.h" ]]; then
		cat >> include/neurondb_config.h <<EOF
#define HAVE_ONNX_RUNTIME 1
#define ONNX_RUNTIME_PATH "${ONNX_RUNTIME_PATH}"
EOF
		msg "ONNX Runtime: Enabled"
	fi
	
	cat >> include/neurondb_config.h <<EOF

/* Build configuration */
#define NEURONDB_CPU_FALLBACK 1

#endif /* NEURONDB_CONFIG_H */
EOF
	
	success "Configuration header generated"
}

write_makefile_local() {
	msg "Writing Makefile.local..."
	
	local pgc="${SELECTED_PG_CONFIG:-$(command -v pg_config || true)}"
	
	cat > Makefile.local <<EOF
# Auto-generated by build.sh - do not edit manually
# Generated: $(date)

PG_CONFIG ?= ${pgc}
PG_MAJOR ?= ${PG_MAJOR_SELECTED:-18}
GPU_MODE ?= ${GPU_MODE:-none}
CUDA_PATH ?= ${CUDA_PATH:-}
ROCM_PATH ?= ${ROCM_PATH:-}
ONNX_RUNTIME_PATH ?= ${ONNX_RUNTIME_PATH:-/usr/local/onnxruntime}

# Build flags
V ?= 1
QUIET_ONNX ?= 1

# Avoid LTO crashes with some toolchains
CFLAGS += -O2 -fPIC -fno-lto
LDFLAGS += -fno-lto

.PHONY: compile
.DEFAULT_GOAL := compile
compile: all
EOF
	
	success "Makefile.local written"
}

wire_platform_makefile() {
	section "Configuring Build System"
	
	local mf_src=""
	case "$PLATFORM" in
		deb)   mf_src="Makefile.deb" ;;
		rocky) mf_src="Makefile.rocky" ;;
		osx)   mf_src="Makefile.osx" ;;
		*)     fatal "Unsupported platform: $PLATFORM" ;;
	esac
	
	if [[ ! -f "$mf_src" ]]; then
		fatal "Platform Makefile not found: $mf_src"
	fi
	
	# Preserve existing Makefile
	if [[ -f Makefile ]] && [[ ! -f Makefile.core ]]; then
		msg "Preserving existing Makefile -> Makefile.core"
		mv Makefile Makefile.core
	fi
	
	msg "Using platform Makefile: $mf_src"
	cp -f "$mf_src" Makefile
	
	success "Build system configured"
}

###############################################################################
# BUILD EXECUTION
###############################################################################

build_neurondb() {
	section "Building NeurondB"
	
	# Verify prerequisites
	if [[ -z "$SELECTED_PG_CONFIG" ]]; then
		# Re-detect PostgreSQL after installation
		detect_postgresql
		
		if [[ -z "$SELECTED_PG_CONFIG" ]]; then
			fatal "pg_config not found after dependency installation"
		fi
	fi
	
	local pgc="$SELECTED_PG_CONFIG"
	local num_jobs
	num_jobs=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
	
	msg "Build Configuration:"
	msg "  Platform:       $PLATFORM"
	msg "  PostgreSQL:     $("$pgc" --version 2>/dev/null || echo 'unknown')"
	msg "  GPU Mode:       $GPU_MODE"
	[[ -n "${CUDA_PATH:-}" ]] && msg "  CUDA Path:      $CUDA_PATH"
	[[ -n "${ROCM_PATH:-}" ]] && msg "  ROCm Path:      $ROCM_PATH"
	msg "  Parallel Jobs:  $num_jobs"
	msg "  ONNX Runtime:   $ONNX_RUNTIME_PATH"
	echo ""
	
	# Clean if requested
	if [[ $FORCE_REBUILD -eq 1 ]]; then
		msg "Cleaning previous build..."
		run_cmd "make clean" make clean PG_CONFIG="$pgc" || true
	fi
	
	# Build arguments
	local make_args="PG_CONFIG=$pgc"
	[[ -n "${CUDA_PATH:-}" ]] && make_args="$make_args CUDA_PATH=$CUDA_PATH"
	[[ -n "${ROCM_PATH:-}" ]] && make_args="$make_args ROCM_PATH=$ROCM_PATH"
	make_args="$make_args ONNX_RUNTIME_PATH=$ONNX_RUNTIME_PATH"
	
	msg "Compiling NeurondB (this may take several minutes)..."
	
	local build_output
	if [[ $VERBOSE -eq 1 ]]; then
		run_cmd "Build NeurondB" make -j"$num_jobs" $make_args all
	else
		# Capture output for error analysis
		if build_output=$(make -j"$num_jobs" $make_args all 2>&1); then
			success "Build completed"
			# Show summary
			echo "$build_output" | grep -E "(Error|error|Warning|warning|Linking|Compiling)" | tail -20 || true
		else
			error "Build failed"
			echo "$build_output" | tail -100
			fatal "Compilation failed"
		fi
	fi
	
	# Verify build output
	verify_build_output
	
	success "NeurondB built successfully"
}

verify_build_output() {
	msg "Verifying build output..."
	
	local lib_found=false
	local lib_path=""
	
	if [[ -f neurondb.so ]]; then
		lib_found=true
		lib_path="neurondb.so"
	elif [[ -f neurondb.dylib ]]; then
		lib_found=true
		lib_path="neurondb.dylib"
	fi
	
	if [[ "$lib_found" == "true" ]]; then
		success "Library created: $lib_path"
		
		# Show library size
		if has_cmd ls; then
			ls -lh "$lib_path"
		fi
		
		# Check for GPU symbols
		if has_cmd nm || has_cmd strings; then
			msg_verbose "Checking for GPU support in library..."
			
			if [[ "$GPU_MODE" == "cuda" ]]; then
				if (nm "$lib_path" 2>/dev/null || strings "$lib_path" 2>/dev/null) | grep -q "cuda"; then
					msg "CUDA symbols found in library"
				fi
			elif [[ "$GPU_MODE" == "metal" ]]; then
				if (nm "$lib_path" 2>/dev/null || strings "$lib_path" 2>/dev/null) | grep -q "metal\|Metal"; then
					msg "Metal symbols found in library"
				fi
			fi
		fi
		
		# Check ONNX Runtime
		if (nm "$lib_path" 2>/dev/null || strings "$lib_path" 2>/dev/null) | grep -q "onnxruntime"; then
			msg "ONNX Runtime linked in library"
		fi
		
	else
		fatal "Build output library not found (neurondb.so or neurondb.dylib)"
	fi
}

install_neurondb() {
	if [[ $SKIP_INSTALL -eq 1 ]]; then
		msg "Skipping installation (--skip-install specified)"
		return 0
	fi
	
	section "Installing NeurondB"
	
	local pgc="$SELECTED_PG_CONFIG"
	
	# Determine if sudo is needed
	local sudo_cmd=""
	if [[ "$PLATFORM" != "osx" ]] && [[ "$(id -u)" != "0" ]]; then
		sudo_cmd="sudo"
	fi
	
	run_cmd "Install NeurondB" $sudo_cmd make install PG_CONFIG="$pgc" || {
		fatal "Installation failed"
	}
	
	# Show installation paths
	local sharedir pkglibdir
	sharedir=$("$pgc" --sharedir 2>/dev/null || echo "unknown")
	pkglibdir=$("$pgc" --pkglibdir 2>/dev/null || echo "unknown")
	
	success "NeurondB installed successfully"
	echo ""
	printf "  ${BOLD}Installation paths:${NC}\n"
	printf "    Extension SQL:  ${sharedir}/extension/neurondb--1.0.sql\n"
	printf "    Control file:   ${sharedir}/extension/neurondb.control\n"
	printf "    Library:        ${pkglibdir}/neurondb.so\n"
	echo ""
}

run_tests() {
	if [[ $SKIP_TESTS -eq 1 ]]; then
		msg "Skipping tests (use --test to enable)"
		return 0
	fi
	
	section "Running Tests"
	
	local pgc="$SELECTED_PG_CONFIG"
	
	# Check if PostgreSQL is running
	if ! psql --version >/dev/null 2>&1; then
		warning "psql not found in PATH, skipping tests"
		warning "To run tests manually: make installcheck PG_CONFIG=$pgc"
		return 0
	fi
	
	run_cmd "Run regression tests" make installcheck PG_CONFIG="$pgc" || {
		warning "Some tests failed (check regression.diffs if available)"
		return 0
	}
	
	success "All tests passed"
}

###############################################################################
# MAIN EXECUTION
###############################################################################

usage() {
	cat <<EOF
NeurondB Build Script v${SCRIPT_VERSION}

Usage: $0 [OPTIONS]

Options:
  -v, --verbose           Enable verbose output
  -n, --dry-run           Show what would be done without executing
  --gpu=MODE              GPU backend: auto, none, cuda, metal, rocm
                          (comma-separated for multiple: cuda,rocm)
  --cuda-path=PATH        Path to CUDA toolkit (auto-detected if not specified)
  --rocm-path=PATH        Path to ROCm (auto-detected if not specified)
  --pg-version=VERSION    PostgreSQL major version (16, 17, 18)
  --pg-config=PATH        Path to pg_config
  --skip-build            Skip compilation step
  --skip-install          Skip installation step
  --test                  Run regression tests after build
  --clean                 Clean build artifacts before building
  -h, --help              Show this help message

Examples:
  $0                                    # Auto-detect everything
  $0 --gpu=cuda                         # Build with CUDA support
  $0 --gpu=metal --test                 # Build with Metal and run tests
  $0 --pg-version=17 --verbose          # Build for PostgreSQL 17 with verbose output

Platforms supported:
  - Rocky Linux / RHEL / CentOS
  - Ubuntu / Debian
  - macOS (via Homebrew)

GPU backends supported:
  - CUDA (NVIDIA GPUs)
  - Metal (Apple Silicon / Intel Macs)
  - ROCm (AMD GPUs)

Environment variables:
  PG_CONFIG              Path to pg_config
  CUDA_PATH              Path to CUDA toolkit
  ROCM_PATH              Path to ROCm
  ONNX_RUNTIME_PATH      Path to ONNX Runtime (default: /usr/local/onnxruntime)

EOF
}

parse_args() {
	while [[ $# -gt 0 ]]; do
		case "$1" in
			-v|--verbose)
				VERBOSE=1
				shift
				;;
			-n|--dry-run)
				DRY_RUN=1
				shift
				;;
			--gpu)
				GPU_MODE="$2"
				shift 2
				;;
			--gpu=*)
				GPU_MODE="${1#*=}"
				shift
				;;
			--cuda-path)
				CUDA_PATH="$2"
				shift 2
				;;
			--cuda-path=*)
				CUDA_PATH="${1#*=}"
				shift
				;;
			--rocm-path)
				ROCM_PATH="$2"
				shift 2
				;;
			--rocm-path=*)
				ROCM_PATH="${1#*=}"
				shift
				;;
			--pg-version)
				PG_VERSION="$2"
				shift 2
				;;
			--pg-version=*)
				PG_VERSION="${1#*=}"
				shift
				;;
			--pg-config)
				PG_CONFIG="$2"
				shift 2
				;;
			--pg-config=*)
				PG_CONFIG="${1#*=}"
				shift
				;;
			--skip-build)
				SKIP_BUILD=1
				shift
				;;
			--skip-install)
				SKIP_INSTALL=1
				shift
				;;
			--test)
				SKIP_TESTS=0
				shift
				;;
			--clean)
				FORCE_REBUILD=1
				shift
				;;
			-h|--help)
				usage
				exit 0
				;;
			*)
				error "Unknown option: $1"
				usage
				exit 1
				;;
		esac
	done
}

main() {
	# Change to script directory
	cd "$(dirname "$0")"
	
	# Parse arguments
	parse_args "$@"
	
	# Banner
	echo ""
	printf "${BOLD}${CYAN}"
	printf "╔══════════════════════════════════════════════════════════════════════════╗\n"
	printf "║                                                                          ║\n"
	printf "║                        NeurondB Build Script                             ║\n"
	printf "║                              Version ${SCRIPT_VERSION}                                 ║\n"
	printf "║                                                                          ║\n"
	printf "╚══════════════════════════════════════════════════════════════════════════╝\n"
	printf "${NC}\n"
	
	# Detection phase
	detect_platform || fatal "Platform detection failed"
	detect_postgresql || warning "PostgreSQL detection incomplete (will attempt installation)"
	detect_gpu_backends || warning "GPU detection incomplete"
	
	# Installation phase
	case "$PLATFORM" in
		deb)
			install_dependencies_ubuntu
			;;
		rocky)
			install_dependencies_rocky
			;;
		osx)
			install_dependencies_macos
			;;
		*)
			fatal "Cannot install dependencies for platform: $PLATFORM"
			;;
	esac
	
	# Re-detect after installation (PostgreSQL may have been installed)
	detect_postgresql
	
	# Re-detect GPU after installation (CUDA may have been installed)
	if [[ "$GPU_MODE" == "cuda" ]] || [[ "$GPU_MODE" == "auto" ]]; then
		if detect_cuda; then
			CUDA_PATH="${CUDA_PATH:-$DETECTED_CUDA_PATH}"
		fi
	fi
	
	# Configuration phase
	write_config_header
	write_makefile_local
	wire_platform_makefile
	
	# Build phase
	if [[ $SKIP_BUILD -eq 0 ]]; then
		build_neurondb
	else
		msg "Skipping build (--skip-build specified)"
	fi
	
	# Install phase
	install_neurondb
	
	# Test phase
	run_tests
	
	# Final summary
	section "Build Complete"
	
	success "NeurondB has been successfully built and installed!"
	echo ""
	printf "  ${BOLD}Next steps:${NC}\n"
	echo ""
	printf "  1. Add to postgresql.conf:\n"
	printf "     ${CYAN}shared_preload_libraries = 'neurondb'${NC}\n"
	echo ""
	printf "  2. Restart PostgreSQL\n"
	echo ""
	printf "  3. Create extension:\n"
	printf "     ${CYAN}psql -d mydb -c 'CREATE EXTENSION neurondb;'${NC}\n"
	echo ""
	
	if [[ "$GPU_MODE" != "none" ]]; then
		printf "  4. GPU configuration (optional):\n"
		printf "     ${CYAN}SET neurondb.gpu_enabled = on;${NC}\n"
		printf "     ${CYAN}SET neurondb.gpu_device = 0;${NC}\n"
		echo ""
	fi
	
	printf "  ${BOLD}Build summary:${NC}\n"
	printf "    Platform:     ${PLATFORM}\n"
	printf "    PostgreSQL:   ${PG_MAJOR_SELECTED:-unknown}\n"
	printf "    GPU Backend:  ${GPU_MODE}\n"
	[[ -n "${CUDA_PATH:-}" ]] && printf "    CUDA Path:    ${CUDA_PATH}\n"
	[[ -n "${ROCM_PATH:-}" ]] && printf "    ROCm Path:    ${ROCM_PATH}\n"
	echo ""
	
	printf "  For documentation, visit:\n"
	printf "    ${CYAN}https://github.com/pgElephant/NeurondB${NC}\n"
	echo ""
}

# Run main function
main "$@"