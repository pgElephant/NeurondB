#!/bin/bash
set -euo pipefail

echo "[init] Applying NeuronDB defaults"

postgresql_conf="${PGDATA}/postgresql.conf"

# Compute mode parameter (0=cpu, 1=gpu, 2=auto, default=2)
compute_mode="${NEURONDB_COMPUTE_MODE:-2}"
gpu_backend_type="${NEURONDB_GPU_BACKEND_TYPE:-0}"
automl_gpu="${NEURONDB_AUTOML_USE_GPU:-off}"

# Normalize accepted values
case "${automl_gpu,,}" in
    on|true|1) automl_gpu=on ;;
    *) automl_gpu=off ;;
esac

if ! grep -q "shared_preload_libraries" "${postgresql_conf}"; then
    cat <<CONF >> "${postgresql_conf}"

# Added by NeuronDB docker image
shared_preload_libraries = 'neurondb'
neurondb.compute_mode = ${compute_mode}
neurondb.gpu_backend_type = ${gpu_backend_type}
neurondb.automl.use_gpu = ${automl_gpu}
CONF
else
    sed -i "s/^shared_preload_libraries.*/shared_preload_libraries = 'neurondb'/g" "${postgresql_conf}"
    
    # Set compute_mode
    if grep -q "^neurondb.compute_mode" "${postgresql_conf}"; then
        sed -i "s/^neurondb.compute_mode.*/neurondb.compute_mode = ${compute_mode}/g" "${postgresql_conf}"
    else
        echo "neurondb.compute_mode = ${compute_mode}" >> "${postgresql_conf}"
    fi
    
    # Set gpu_backend_type
    if grep -q "^neurondb.gpu_backend_type" "${postgresql_conf}"; then
        sed -i "s/^neurondb.gpu_backend_type.*/neurondb.gpu_backend_type = ${gpu_backend_type}/g" "${postgresql_conf}"
    else
        echo "neurondb.gpu_backend_type = ${gpu_backend_type}" >> "${postgresql_conf}"
    fi
    
    if grep -q "^neurondb.automl.use_gpu" "${postgresql_conf}"; then
        sed -i "s/^neurondb.automl.use_gpu.*/neurondb.automl.use_gpu = ${automl_gpu}/g" "${postgresql_conf}"
    else
        echo "neurondb.automl.use_gpu = ${automl_gpu}" >> "${postgresql_conf}"
    fi
fi

