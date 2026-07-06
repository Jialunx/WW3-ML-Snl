# Global example — realistic 1-degree global run

Run the ML `S_nl` surrogate on a realistic global grid (1 degree, 360 x 181),
forced by ERA5 winds. The spectral grid is 24 directions x 40 frequencies,
as the trained networks require. Use ML or ML-Lite here (the depth-scaled
networks); ML-FiLM is for idealized finite-depth cases, not global runs.

## Files
- `ww3_grid.nml`, `namelists_Global-real.nml`  grid + spectral definition
- `Global-real.depth`                           1-degree global bathymetry
- `ww3_prnc.nml`                                wind-forcing conversion config
- `ww3_shel.nml`                                run config (period, output)
- `points.list`                                output points
- `download_era5_wind.py`                       fetch + remap ERA5 winds (needs a CDS account)
- `run_global.sh`                              build-and-run helper

## Steps

Build the package first (top-level README quick start), then from this folder:

```sh
export ORT=/path/to/onnxruntime          # same ONNX Runtime used to build

# 1. get the wind (free Copernicus CDS account required)
pip install cdsapi xarray netcdf4
python download_era5_wind.py             # -> wind_1deg_global_20250101to15.nc

# 2. run with ML-Lite (or MODEL=unet_faster_24x40_base32_deep.onnx for ML)
bash run_global.sh
```

Or run the stages manually:
```sh
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
