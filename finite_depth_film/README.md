# ML-FiLM: finite-depth (depth-conditioned) S_nl build

The default build in the top-level `README.md` uses the depth-scaled surrogate
(ML / ML-Lite), which takes the spectrum as its only input. The finite-depth
surrogate **ML-FiLM** additionally conditions on water depth (through the
frequency-resolved group velocity and `k_p d`), so it uses a different module
and a two-input ONNX Runtime wrapper.

## Files

- `w3snl6md.F90`: the NL6 module for the depth-conditioned surrogate. It builds
  the input spectrum, supplies the scalar depth, calls the two-input forward,
  and rescales the result back to WW3 action units. The full inference recipe
  (peak centering, normalization, the group-velocity condition vector, the
  depth limiter, and the U-Net forward) is baked into the exported ONNX graph.

The two-input ONNX Runtime wrapper (`ort_wrapper_film.{c,h}`) already ships in
`model/src/ort/` alongside the depth-scaled wrapper. CMake selects between them
with `-DWW3_SNL_VARIANT`, so no wrapper files need to be moved.

## How to build

Copy the FiLM module over the active one, then configure with the `film`
variant:

```sh
cp finite_depth_film/w3snl6md.F90 model/src/w3snl6md.F90

cmake -S . -B build_film \
      -DSWITCH=NL6_ML \
      -DORT_ROOT=/path/to/onnxruntime \
      -DWW3_SNL_VARIANT=film
cmake --build build_film -j
```

`-DWW3_SNL_VARIANT=film` makes CMake compile `ort_wrapper_film.c` (two-input)
instead of `ort_wrapper.c` (single-input). Keep a clean checkout (or
`git stash`) if you want to switch back to the depth-scaled build afterward.

## Run

```sh
export WW3_SNL_ONNX_MODEL=ml_models/cond_unet_film_24x40.onnx
```

The model path defaults to the bundled FiLM weights if the environment variable
is unset. As with the depth-scaled build, run WW3 from a directory containing
`ml_models/`, or give an absolute path.
