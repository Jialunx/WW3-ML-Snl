#!/usr/bin/env bash
# One-command build + run of the ML S_nl surrogate on the global example:
# a 1-hour, 1-degree global run with ML-Lite and a bundled ERA5 wind field.
#
# Requires: Fortran + C compiler, CMake, MPI, NetCDF, curl.
# Usage:
#   bash quickstart.sh                                        # ML-Lite (default)
#   MODEL=unet_faster_24x40_base32_deep.onnx bash quickstart.sh   # ML
#   ORT=/path/to/onnxruntime bash quickstart.sh              # use an existing ONNX Runtime
#   NP=8 bash quickstart.sh                                  # MPI ranks (default 4)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
ORT_VER=${ORT_VER:-1.20.1}
MODEL=${MODEL:-unet_faster_24x40_base16.onnx}
NP=${NP:-4}

# 1. ONNX Runtime (download a prebuilt CPU build unless ORT is already provided)
if [ -z "${ORT:-}" ] || [ ! -f "${ORT}/lib/libonnxruntime.so" ]; then
  ORT="$ROOT/onnxruntime-linux-x64-${ORT_VER}"
  if [ ! -f "$ORT/lib/libonnxruntime.so" ]; then
    echo ">> downloading ONNX Runtime ${ORT_VER}"
    curl -L "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-${ORT_VER}.tgz" | tar xz
  fi
fi
export ORT

# 2. build with the NL6 ML surrogate
echo ">> building (SWITCH=NL6_ML)"
cmake -S . -B build -DSWITCH=NL6_ML -DORT_ROOT="$ORT"
cmake --build build -j

# 3. run the global example: 1 hour, 1-degree grid, bundled ERA5 wind
cd example_global
export LD_LIBRARY_PATH="$ORT/lib:${LD_LIBRARY_PATH:-}"
export WW3_SNL_ONNX_MODEL="$ROOT/ml_models/$MODEL"
echo ">> running the 1-hour global example with $MODEL"
mpirun -np 1  "$ROOT/build/bin/ww3_grid"     # -> mod_def.ww3
mpirun -np 1  "$ROOT/build/bin/ww3_prnc"     # ERA5 wind -> wind.ww3
mpirun -np "$NP" "$ROOT/build/bin/ww3_shel"  # cold-starts from calm; ML computes S_nl
echo ">> done. output is in $(pwd)"
