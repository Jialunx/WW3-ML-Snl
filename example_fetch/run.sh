#!/usr/bin/env bash
# Build (if needed) and run the local basin example with ML-Lite.
# Just:  cd example && bash run.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/.." && pwd)

# 0. prerequisites
for t in gfortran cmake mpirun curl; do command -v "$t" >/dev/null || {
  echo "Missing '$t'. Install prerequisites first:"
  echo "  sudo apt install -y build-essential gfortran cmake libopenmpi-dev libnetcdf-dev libnetcdff-dev curl"
  exit 1; }; done

# 1. ONNX Runtime (download if absent)
ORT_VER=${ORT_VER:-1.20.1}
ORT=${ORT:-$ROOT/onnxruntime-linux-x64-$ORT_VER}
if [ ! -f "$ORT/lib/libonnxruntime.so" ]; then
  echo ">> downloading ONNX Runtime $ORT_VER"
  ( cd "$ROOT" && curl -L "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-${ORT_VER}.tgz" | tar xz )
  ORT="$ROOT/onnxruntime-linux-x64-$ORT_VER"
fi

# 2. build if needed
if [ ! -x "$ROOT/build/bin/ww3_shel" ]; then
  echo ">> building"
  cmake -S "$ROOT" -B "$ROOT/build" -DSWITCH=NL6_ML -DORT_ROOT="$ORT"
  cmake --build "$ROOT/build" -j
fi

# 3. run with ML-Lite (MODEL=unet_faster_24x40_base32_deep.onnx for ML)
cd "$HERE"
MODEL=${MODEL:-unet_faster_24x40_base16.onnx}
export LD_LIBRARY_PATH="$ORT/lib:${LD_LIBRARY_PATH:-}"
export WW3_SNL_ONNX_MODEL="$ROOT/ml_models/$MODEL"
mpirun -np 1 "$ROOT/build/bin/ww3_grid"
mpirun -np 1 "$ROOT/build/bin/ww3_strt"
mpirun -np 1 "$ROOT/build/bin/ww3_shel"
echo ">> done. outputs are in $HERE"
