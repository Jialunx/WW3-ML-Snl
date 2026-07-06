# Global example — realistic 1-degree global run

Run the ML `S_nl` surrogate on a realistic global grid (1 degree, 360 x 181),
forced by ERA5 winds. The spectral grid is 24 directions x 40 frequencies,
as the trained networks require. Use ML or ML-Lite here (the depth-scaled
networks); ML-FiLM is for idealized finite-depth cases, not global runs.

## Files
- `ww3_grid.nml`, `namelists_Global-real.nml`  grid + spectral definition
- `Global-real.depth`                           1-degree global bathymetry
- `wind_example.nc`                             real ERA5 10 m wind, 2025-01-01 (24 h) — bundled
- `ww3_prnc.nml`, `ww3_shel.nml`, `points.list` run configuration
- `download_era5_wind.py`                       (optional) fetch the full multi-day ERA5 wind
- `run_global.sh`                              build-and-run helper

## Run it

A real ERA5 wind for 2025-01-01 (24 h, the field used in the paper) is bundled
as `wind_example.nc`, so this runs out of the box — no account or download:

```sh
sudo apt install -y build-essential gfortran cmake libopenmpi-dev libnetcdf-dev libnetcdff-dev curl git
git clone https://github.com/Jialunx/WW3-ML-Snl.git && cd WW3-ML-Snl/example_global
bash run_global.sh                              # MODEL=unet_faster_24x40_base32_deep.onnx bash run_global.sh  for ML
```

`run_global.sh` downloads ONNX Runtime, builds if needed, then runs
`ww3_grid -> ww3_prnc -> ww3_shel` with the bundled wind. It cold-starts from
calm and spins up under the wind; field output is written to `ww3.*.nc`.

### Full paper period (optional)

The bundled wind covers one day. For the full 14-day period, download the real
ERA5 wind (free Copernicus CDS account; put your key in `~/.cdsapirc`):
```sh
python3 -m venv ~/cds-venv && ~/cds-venv/bin/pip install cdsapi xarray netcdf4
~/cds-venv/bin/python download_era5_wind.py     # -> wind_1deg_global_20250101to15.nc
```
Then set `FILE%FILENAME` in `ww3_prnc.nml` to that file, widen
`FORCING%TIMESTOP` and `DOMAIN%STOP`, and rerun.

`run_global.sh` downloads ONNX Runtime, builds if not already built, then runs
`ww3_grid -> ww3_prnc -> ww3_shel`. It cold-starts from calm and spins up under
the wind; field output is written to `ww3.*.nc`.

Already cloned? `cd example_global && ~/cds-venv/bin/python download_era5_wind.py && bash run_global.sh`.
If the clone says the folder already exists, update it instead: `cd WW3-ML-Snl && git pull`.

Or run the stages manually. You must **build first**, else the `mpirun` lines
fail with "could not access or execute ../build/bin/ww3_grid".
```sh
# 1. build (from the repo root, one level up)
cd ..
ORT_VER=1.20.1
curl -L https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-${ORT_VER}.tgz | tar xz
export ORT=$PWD/onnxruntime-linux-x64-${ORT_VER}
cmake -S . -B build -DSWITCH=NL6_ML -DORT_ROOT=$ORT
cmake --build build -j          # wait for "Built target ww3_shel"

# 2. run (from this folder; wind must already be downloaded)
cd example_global
export LD_LIBRARY_PATH=$ORT/lib:$LD_LIBRARY_PATH
export WW3_SNL_ONNX_MODEL=$PWD/../ml_models/unet_faster_24x40_base16.onnx
mpirun -np 1 ../build/bin/ww3_grid     # -> mod_def.ww3
mpirun -np 1 ../build/bin/ww3_prnc     # ERA5 wind -> wind.ww3
mpirun -np 4 ../build/bin/ww3_shel     # cold-starts from calm; ML computes S_nl
```

The run cold-starts from a calm sea and spins up under the wind. Field output is
written to `ww3.*.nc` (convert the raw `out_grd.ww3` with `ww3_ounf` if needed).

## Choosing the period

`ww3_shel.nml` sets the simulated period with `DOMAIN%START` / `DOMAIN%STOP`:
- **1 hour** (quick test):  `DOMAIN%STOP = '20250101 010000'`
- **Full wind file** (~14 days): `DOMAIN%STOP = '20250115 000000'`
- **1 month**: download a month of wind (edit `download_era5_wind.py`) and set
  `DOMAIN%STOP = '20250201 000000'`.

The wind file must cover the whole run period.

## Notes
- Cost scales with grid size: a 1-hour global step ran in about 2.5 minutes on
  4 MPI ranks in testing. Use more ranks (`mpirun -np N`) for longer runs.
- Switch model by changing `WW3_SNL_ONNX_MODEL`; no rebuild needed.
