# WW3-ML-Snl: machine-learning surrogate for the nonlinear four-wave interaction in WAVEWATCH III

This package couples a machine-learning surrogate for the nonlinear four-wave
interaction (`S_nl`) into **WAVEWATCH III v7.14**. The surrogate replaces the
Discrete Interaction Approximation (DIA) as a new source-term option (`NL6`)
and is evaluated at run time through the ONNX Runtime C API.

Three trained surrogates are bundled:

| Model | File (`ml_models/`) | Description |
|-------|---------------------|-------------|
| ML       | `unet_faster_24x40_base32_deep.onnx` | Base U-Net (32/64/128 ch), depth-scaled, tuned for accuracy |
| ML-Lite  | `unet_faster_24x40_base16.onnx`      | Lightweight U-Net (16/32 ch), tuned for fast inference |
| ML-FiLM  | `cond_unet_film_24x40.onnx`          | Depth-conditioned U-Net, reproduces the finite-depth interaction |

ML and ML-Lite emulate the depth-scaled Webb-Resio-Tracy (WRT) transfer and
share one single-input module. ML-FiLM emulates the finite-depth WRT directly
and uses a two-input (spectrum + depth) module, provided in
`finite_depth_film/` as a drop-in variant.

## Quick start

Requires a Fortran and C compiler, CMake, MPI, and NetCDF.

**One command** (clone, then run the helper script — downloads ONNX Runtime,
builds, and runs the example with ML-Lite):

```sh
git clone https://github.com/Jialunx/WW3-ML-Snl.git
cd WW3-ML-Snl
bash quickstart.sh          # ML-Lite;  MODEL=unet_faster_24x40_base32_deep.onnx bash quickstart.sh  for ML
```

Or run the steps manually:

```sh
# 1. get the code
git clone https://github.com/Jialunx/WW3-ML-Snl.git
cd WW3-ML-Snl

# 2. get ONNX Runtime (prebuilt, CPU)
ORT_VER=1.20.1
curl -L https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-${ORT_VER}.tgz | tar xz
export ORT=$PWD/onnxruntime-linux-x64-${ORT_VER}

# 3. build with the ML surrogate (NL6)
cmake -S . -B build -DSWITCH=NL6_ML -DORT_ROOT=$ORT
cmake --build build -j

# 4. run the example with ML-Lite
cd example
export LD_LIBRARY_PATH=$ORT/lib:$LD_LIBRARY_PATH
export WW3_SNL_ONNX_MODEL=$PWD/../ml_models/unet_faster_24x40_base16.onnx
mpirun -np 1 ../build/bin/ww3_grid
mpirun -np 1 ../build/bin/ww3_strt
mpirun -np 1 ../build/bin/ww3_shel
```

A successful run prints `[ort_wrapper] rank 0: ort_forward calls = ...` and
`End of program`. For ML use `unet_faster_24x40_base32_deep.onnx`; the ML-FiLM
model needs the finite-depth build (see `finite_depth_film/`).

## Repository layout

```
WW3-ML-Snl-v1.0/
  CMakeLists.txt, cmake/, model/, VERSION   WAVEWATCH III v7.14 source tree
  model/src/w3snl6md.F90                     ML S_nl module (NL6), depth-scaled (ML / ML-Lite)
  model/src/ort/ort_wrapper.{c,h}            ONNX Runtime C wrapper (single input)
  model/bin/switch_NL6_ML                    compile switch enabling NL6
  ml_models/                                 the three trained ONNX weights + training notebooks
  finite_depth_film/                         drop-in files for the ML-FiLM build
  LICENSE.md                                 WAVEWATCH III license
```

## Dependencies

- A Fortran and C compiler (GNU or Intel), CMake >= 3.19
- MPI (the bundled switch uses `DIST MPI`)
- NetCDF (Fortran + C)
- [ONNX Runtime](https://onnxruntime.ai) (C/C++ build). A build providing the
  DNNL execution provider is recommended for CPU performance. Set `ORT_ROOT`
  to the install directory (it must contain `include/` and `lib/`).

## Build (ML / ML-Lite, depth-scaled)

```sh
cmake -S . -B build \
      -DSWITCH=NL6_ML \
      -DORT_ROOT=/path/to/onnxruntime
cmake --build build -j
```

This produces the standard WW3 programs (`ww3_grid`, `ww3_shel`, `ww3_ounf`,
...) linked against the ML `S_nl` module. The `NL6` token in the switch selects
the surrogate as the nonlinear source term.

## Selecting a model at run time

The active `.onnx` file is chosen by the environment variable
`WW3_SNL_ONNX_MODEL`. If unset, the module falls back to the bundled default
(`ml_models/unet_faster_24x40_base32_deep.onnx`). Paths are resolved relative to
the run directory, so run WW3 from a directory that contains `ml_models/`, or
give an absolute path.

```sh
# ML (base, depth-scaled)   -- default
export WW3_SNL_ONNX_MODEL=ml_models/unet_faster_24x40_base32_deep.onnx

# ML-Lite (lightweight, depth-scaled)
export WW3_SNL_ONNX_MODEL=ml_models/unet_faster_24x40_base16.onnx
```

ML and ML-Lite share the same compiled binary. Only the environment variable
changes.

## Build for the finite-depth surrogate (ML-FiLM)

ML-FiLM uses a different module and wrapper because it feeds water depth to the
network in addition to the spectrum. Swap the two source files in, then rebuild:

```sh
cp finite_depth_film/w3snl6md.F90 model/src/w3snl6md.F90
cmake -S . -B build_film -DSWITCH=NL6_ML -DORT_ROOT=/path/to/onnxruntime -DWW3_SNL_VARIANT=film
cmake --build build_film -j
export WW3_SNL_ONNX_MODEL=ml_models/cond_unet_film_24x40.onnx
```

The two-input FiLM wrapper ships in `model/src/ort/`; `-DWW3_SNL_VARIANT=film`
selects it. Only the module (`w3snl6md.F90`) has to be copied in.

See `finite_depth_film/README.md` for details. The trained models operate on a
24-direction by 40-frequency spectral grid, matching the configuration used in
the paper.

## Training

The notebooks used to train the bundled weights are in `ml_models/`:

- `train_ML_and_MLLite.ipynb` trains the depth-scaled surrogates (ML, `base_ch=32`;
  ML-Lite, `base_ch=16`) on deep-water WRT labels.
- `train_ML_FiLM.ipynb` trains the depth-conditioned ML-FiLM surrogate on
  finite-depth WRT labels.

Each notebook covers data loading, preprocessing and normalization, and training.
The training data (energy-density spectra, `S_nl` targets, and depth, as CSV files)
is not distributed with this package. Set the data directory variables at the top
of each notebook to point at your own copy.

## Attribution

WAVEWATCH III is developed by NOAA/NCEP and the WAVEWATCH III Development Group
and is distributed under the license in `LICENSE.md`. This package adds the
`NL6` machine-learning `S_nl` source term (`model/src/w3snl6md.F90`, the ONNX
Runtime wrapper, and the trained weights) on top of an unmodified WW3 v7.14
tree. All other source files are as distributed by WW3.

## Citation

If you use this software, please cite the accompanying paper by
Chen, J., Adcock, T. A. A., Liu, Q., Clark, R., & Tang, T.
