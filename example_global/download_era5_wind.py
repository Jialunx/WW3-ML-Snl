#!/usr/bin/env python
"""Download ERA5 10 m winds and remap them to the WW3 1-degree global grid.

Requires a (free) Copernicus CDS account and the cdsapi package:
    pip install cdsapi xarray netcdf4
    # put your CDS key in ~/.cdsapirc  (see https://cds.climate.copernicus.eu/api-how-to)

Edit YEAR / MONTH / DAYS below for the period you want (default: 2025-01-01..15).
Output (in this folder): wind_1deg_global_20250101to15.nc  -> feed to ww3_prnc.
"""
import cdsapi, numpy as np, xarray as xr

YEAR, MONTH, DAYS = "2025", "01", list(range(1, 16))   # 2025-01-01 .. 2025-01-15
era5_file = "era5_uv10_raw.nc"
out_file  = "wind_1deg_global_20250101to15.nc"

req = {
    "product_type": "reanalysis",
    "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
    "year": YEAR, "month": MONTH,
    "day": [f"{d:02d}" for d in DAYS],
    "time": [f"{h:02d}:00" for h in range(24)],
    "data_format": "netcdf", "download_format": "unarchived",
}
cdsapi.Client().retrieve("reanalysis-era5-single-levels", req, era5_file)

ds = xr.open_dataset(era5_file)
tname = "time" if "time" in ds.coords else "valid_time"
for d in ("number", "expver"):
    if d in ds.dims:
        ds = ds.isel({d: 0})
if ds["longitude"].min() < 0:                      # -> 0..359
    ds = ds.assign_coords(longitude=(ds["longitude"] % 360)).sortby("longitude")
lon_t, lat_t = np.arange(0., 360., 1.), np.arange(-90., 91., 1.)
u = ds["u10"].interp(longitude=lon_t, latitude=lat_t, method="linear")
v = ds["v10"].interp(longitude=lon_t, latitude=lat_t, method="linear")
xr.Dataset(
    {"u10": ((tname, "latitude", "longitude"), u.values),
     "v10": ((tname, "latitude", "longitude"), v.values)},
    coords={tname: ds[tname].values, "latitude": lat_t, "longitude": lon_t},
).rename({tname: "time"}).to_netcdf(out_file)
print("saved", out_file)
