#!/usr/bin/env bash
# Run the global case with the ML surrogate. Build the package first
# (see the top-level README quick start), then from this folder:
#   1. python download_era5_wind.py      # get wind_1deg_global_*.nc  (needs a CDS account)
#   2. bash run_global.sh                # ML-Lite by default
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
ROOT=$(cd "$HERE/.." && pwd)
BIN=${BIN:-$ROOT/build/bin}
MODEL=${MODEL:-unet_faster_24x40_base16.onnx}
NP=${NP:-4}
: "${ORT:?set ORT to your ONNX Runtime dir (the one used to build)}"
[ -f wind_1deg_global_20250101to15.nc ] || { echo "wind file missing - run: python download_era5_wind.py"; exit 1; }
export LD_LIBRARY_PATH="$ORT/lib:${LD_LIBRARY_PATH:-}"
export WW3_SNL_ONNX_MODEL="$ROOT/ml_models/$MODEL"
mpirun -np 1  "$BIN/ww3_grid"          # -> mod_def.ww3
mpirun -np 1  "$BIN/ww3_prnc"          # ERA5 wind -> wind.ww3
mpirun -np $NP "$BIN/ww3_shel"         # cold-starts from calm; ML computes S_nl
echo "done. field output: ww3.*.nc"
