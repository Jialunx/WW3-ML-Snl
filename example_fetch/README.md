# Example case: fetch-limited basin (24x40)

Self-contained case for the ML `S_nl` surrogate: a shallow (8 m) basin
(122 x 5 cells) forced from rest by a uniform 20 m/s wind for 1 hour. Spectral
grid is 24 x 40 (`f0 = 0.03453 Hz`, ratio 1.1), fixed by the networks.

## Run

```sh
sudo apt install -y build-essential gfortran cmake libopenmpi-dev libnetcdf-dev libnetcdff-dev curl git
git clone https://github.com/Jialunx/WW3-ML-Snl.git && cd WW3-ML-Snl/example_fetch && bash run.sh
```

`run.sh` gets ONNX Runtime, builds if needed, and runs ML-Lite. Ends with
`End of program` and writes `ww3.*` output here. For ML:
`MODEL=unet_faster_24x40_base32_deep.onnx bash run.sh`. Already cloned:
`cd example_fetch && bash run.sh`.

## Files
- `ww3_grid.inp`: grid + spectral definition
- `basin8.bot`, `basin8.mask`: bathymetry and mask
- `ww3_strt.inp`: initial condition
- `ww3_shel.nml`: 1-hour run, homogeneous 20 m/s wind
- `points.list`: output points

## Manual build and run

```sh
# build (repo root)
cd ..
ORT_VER=1.20.1
curl -L https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-${ORT_VER}.tgz | tar xz
export ORT=$PWD/onnxruntime-linux-x64-${ORT_VER}
cmake -S . -B build -DSWITCH=NL6_ML -DORT_ROOT=$ORT && cmake --build build -j

# run (this folder)
cd example_fetch
export LD_LIBRARY_PATH=$ORT/lib:$LD_LIBRARY_PATH
export WW3_SNL_ONNX_MODEL=$PWD/../ml_models/unet_faster_24x40_base16.onnx
mpirun -np 1 ../build/bin/ww3_grid
mpirun -np 1 ../build/bin/ww3_strt
mpirun -np 1 ../build/bin/ww3_shel
```

The ST4 table (`ST4TABUHF2.bin`) is generated on the first run. The FiLM model
needs the finite-depth build (see `../finite_depth_film/`).

## Output

Raw output (`out_grd.ww3`, `out_pnt.ww3`) is written here. Convert to NetCDF
(keep `WW3_SNL_ONNX_MODEL` and `LD_LIBRARY_PATH` set, as `ww3_ounf`/`ww3_ounp`
also load the surrogate):

```sh
mpirun -np 1 ../build/bin/ww3_ounf                 # bulk fields -> ww3.*.nc (HS, T02, ...)

cp ww3_ounp_src.nml ww3_ounp.nml
mpirun -np 1 ../build/bin/ww3_ounp                 # source terms -> ww3.*_src.nc
```

`ww3.*_src.nc` holds `snl` (the ML `S_nl`), `sin`, `sds`, `stt`, and `efth`,
each as `F(f,theta)` at the `points.list` stations.
