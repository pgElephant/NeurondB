#!/bin/bash
set -euo pipefail

echo "[init] Applying NeuronDB defaults"

postgresql_conf="${PGDATA}/postgresql.conf"

gpu_enabled="${NEURONDB_GPU_ENABLED:-off}"
automl_gpu="${NEURONDB_AUTOML_USE_GPU:-off}"

# Normalize accepted values
case "${gpu_enabled,,}" in
    on|true|1) gpu_enabled=on ;;
    *) gpu_enabled=off ;;
esac

case "${automl_gpu,,}" in
    on|true|1) automl_gpu=on ;;
    *) automl_gpu=off ;;
esac

if ! grep -q "shared_preload_libraries" "${postgresql_conf}"; then
    cat <<CONF >> "${postgresql_conf}"

# Added by NeuronDB docker image
shared_preload_libraries = 'neurondb'
neurondb.gpu_enabled = ${gpu_enabled}
neurondb.automl.use_gpu = ${automl_gpu}
CONF
else
    sed -i "s/^shared_preload_libraries.*/shared_preload_libraries = 'neurondb'/g" "${postgresql_conf}"
    if grep -q "^neurondb.gpu_enabled" "${postgresql_conf}"; then
        sed -i "s/^neurondb.gpu_enabled.*/neurondb.gpu_enabled = ${gpu_enabled}/g" "${postgresql_conf}"
    else
        echo "neurondb.gpu_enabled = ${gpu_enabled}" >> "${postgresql_conf}"
    fi
    if grep -q "^neurondb.automl.use_gpu" "${postgresql_conf}"; then
        sed -i "s/^neurondb.automl.use_gpu.*/neurondb.automl.use_gpu = ${automl_gpu}/g" "${postgresql_conf}"
    else
        echo "neurondb.automl.use_gpu = ${automl_gpu}" >> "${postgresql_conf}"
    fi
fi

