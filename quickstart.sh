#!/usr/bin/env bash
# One-command build + run of the ML S_nl surrogate on the bundled example case.
#
# Requires: Fortran + C compiler, CMake, MPI, NetCDF, curl.
# Usage:
#   bash quickstart.sh                # ML-Lite (default)
#   MODEL=unet_faster_24x40_base32_deep.onnx bash quickstart.sh   # ML
#   ORT=/path/to/onnxruntime bash quickstart.sh                   # use an existing ONNX Runtime
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
ORT_VER=${ORT_VER:-1.20.1}
MODEL=${MODEL:-unet_faster_24x40_base16.onnx}

# 1. ONNX Runtime (download a prebuilt CPU build unless ORT is already provided)
if [ -z "${ORT:-}" ] || [ ! -f "${ORT}/lib/libonnxruntime.so" ]; then
  ORT="$ROOT/onnxruntime-linux-x64-${ORT_VER}"
  if [ ! -f "$ORT/lib/libonnxruntime.so" ]; then
    echo ">> downloading ONNX Runtime ${ORT_VER}"
    curl -L "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-${ORT_VER}.tgz" | tar xz
  fi
fi
export ORT
echo ">> ONNX Runtime: $ORT"

# 2. build with the NL6 ML surrogate
echo ">> building (SWITCH=NL6_ML)"
cmake -S . -B build -DSWITCH=NL6_ML -DORT_ROOT="$ORT"
cmake --build build -j

# 3. run the example with the chosen model
cd example
export LD_LIBRARY_PATH="$ORT/lib:${LD_LIBRARY_PATH:-}"
export WW3_SNL_ONNX_MODEL="$ROOT/ml_models/$MODEL"
echo ">> running example with $MODEL"
mpirun -np 1 ../build/bin/ww3_grid
mpirun -np 1 ../build/bin/ww3_strt
mpirun -np 1 ../build/bin/ww3_shel
echo ">> done. outputs are in $(pwd)"
