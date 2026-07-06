# Example case — fetch-limited basin (24x40)

A small self-contained case for the ML `S_nl` surrogate: a shallow (8 m)
rectangular basin (122 x 5 cells) forced from rest by a uniform 20 m/s wind,
run for 1 hour. The spectral grid is 24 directions x 40 frequencies
(`f0 = 0.03453 Hz`, ratio 1.1), which the trained networks require.

## Run it

Copy-paste this whole block (installs prerequisites, gets the code, builds, and
runs ML-Lite — no navigating or path editing needed):

```sh
sudo apt install -y build-essential gfortran cmake libopenmpi-dev libnetcdf-dev libnetcdff-dev curl git
git clone https://github.com/Jialunx/WW3-ML-Snl.git && cd WW3-ML-Snl/example && bash run.sh
```

`run.sh` downloads ONNX Runtime, builds the model (if not already built), and
runs the case with ML-Lite. Ends with `End of program` and writes `ww3.*`
outputs here. Use `MODEL=unet_faster_24x40_base32_deep.onnx bash run.sh` for ML.

Already have the repo? Just `cd example && bash run.sh`.

Files:
- `ww3_grid.inp`  grid + spectral definition
- `basin8.bot`, `basin8.mask`  bathymetry and mask
- `ww3_strt.inp`  initial condition
- `ww3_shel.nml`  1-hour run, homogeneous 20 m/s wind
- `points.list`   output points

Run (from this folder, after building with `-DSWITCH=NL6_ML -DORT_ROOT=...`):

```sh
export LD_LIBRARY_PATH=$ORT/lib:$LD_LIBRARY_PATH
export WW3_SNL_ONNX_MODEL=$PWD/../ml_models/unet_faster_24x40_base16.onnx   # ML-Lite
mpirun -np 1 ../build/bin/ww3_grid
mpirun -np 1 ../build/bin/ww3_strt
mpirun -np 1 ../build/bin/ww3_shel
```

The ST4 wind-input table (`ST4TABUHF2.bin`) is generated automatically on the
first run. Switch model by changing `WW3_SNL_ONNX_MODEL` (…base32_deep.onnx for
ML; the FiLM model needs the finite-depth build, see `../finite_depth_film/`).
