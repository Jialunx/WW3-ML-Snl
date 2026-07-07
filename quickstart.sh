#!/usr/bin/env bash
# One command: build (if needed) and run the 1-hour global example with ML-Lite.
#
# This is a thin wrapper around example_global/run_global.sh, which downloads
# ONNX Runtime, builds, fetches the warm-start initial condition (the paper's
# ERA5 spin-up state), and runs ww3_grid -> ww3_prnc -> ww3_shel.
#
# Requires: Fortran + C compiler, CMake, MPI, NetCDF, curl.
#   bash quickstart.sh                                       # ML-Lite, warm start (default)
#   MODEL=unet_faster_24x40_base32_deep.onnx bash quickstart.sh   # ML
#   WARM=0 bash quickstart.sh                                # cold start from calm
#   NP=8 bash quickstart.sh                                  # MPI ranks (default 4)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
exec bash "$ROOT/example_global/run_global.sh" "$@"
