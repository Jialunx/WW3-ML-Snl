/* ort_wrapper.c -- minimal Fortran-friendly wrapper for ONNX Runtime.
 *
 * Owns one process-wide InferenceSession.  Single-input / single-output
 * FP32 4-D inference only, matching what w3snl6md.F90 needs.
 *
 * The expected ONNX model is unet_faster_24x40_base16.onnx (or any model
 * with the same I/O signature; see ort_init).
 */

#include "ort_wrapper.h"
#include <onnxruntime_c_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* dnnl_provider_factory.h pulls in <memory> (C++), so we forward-declare
 * the DNNL provider entry point here instead.  Symbol is exported by
 * libonnxruntime.so when ORT is built with --use_dnnl.                  */
extern OrtStatus *OrtSessionOptionsAppendExecutionProvider_Dnnl(
    OrtSessionOptions *options, int use_arena);

/* The exported ONNX model uses these node names.  See export step:
 *   torch.onnx.export(..., input_names=['x'], output_names=['snl'], ...) */
static const char *kInputName  = "x";
static const char *kOutputName = "snl";

static const OrtApi    *g_ort     = NULL;
static OrtEnv          *g_env     = NULL;
static OrtSession      *g_session = NULL;
static OrtMemoryInfo   *g_meminfo = NULL;
static OrtValue        *g_in_val  = NULL;
static OrtValue        *g_out_val = NULL;
static int              g_dims_h  = 0;
static int              g_dims_w  = 0;
/* Per-rank counter of ort_forward calls; printed by ort_finalize. */
static long long        g_n_calls = 0;

#define CHECK_STATUS(status, where)                                     \
    do {                                                                \
        if (status != NULL) {                                           \
            const char *msg = g_ort->GetErrorMessage(status);           \
            fprintf(stderr, "[ort_wrapper] %s: %s\n", where, msg);      \
            g_ort->ReleaseStatus(status);                               \
            return 1;                                                   \
        }                                                               \
    } while (0)

/* atexit handler -- prints the call count even when Fortran never invokes
 * ML_SNL_FINALIZE.  Registered once from ort_init.                       */
static void ort_dump_counts_atexit(void)
{
    const char *rk = getenv("OMPI_COMM_WORLD_RANK");
    if (!rk) rk = getenv("PMI_RANK");
    if (!rk) rk = "?";
    fprintf(stderr, "[ort_wrapper] rank %s: ort_forward calls = %lld\n",
            rk, g_n_calls);
    fflush(stderr);
}

int ort_init(const char *model_path)
{
    if (g_session) return 0;  /* idempotent */
    atexit(ort_dump_counts_atexit);

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "[ort_wrapper] OrtGetApiBase()->GetApi() returned NULL\n");
        return 1;
    }

    OrtStatus *st;

    st = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ww3_ort", &g_env);
    CHECK_STATUS(st, "CreateEnv");

    OrtSessionOptions *so;
    st = g_ort->CreateSessionOptions(&so);
    CHECK_STATUS(st, "CreateSessionOptions");

    /* Mirror libtorch single-thread-per-rank behaviour: one rank = one
     * inference thread.  Caller controls MPI parallelism instead.       */
    st = g_ort->SetIntraOpNumThreads(so, 1);
    CHECK_STATUS(st, "SetIntraOpNumThreads");
    st = g_ort->SetInterOpNumThreads(so, 1);
    CHECK_STATUS(st, "SetInterOpNumThreads");
    st = g_ort->SetSessionExecutionMode(so, ORT_SEQUENTIAL);
    CHECK_STATUS(st, "SetSessionExecutionMode");
    st = g_ort->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL);
    CHECK_STATUS(st, "SetSessionGraphOptimizationLevel");

    /* DNNL EP was tried here but empirically didn't help: most ops
     * (CircularPad2D, replication-pad) fall back to CPU EP anyway, and
     * the fallback shuffle between EPs adds per-call overhead.  Stick
     * with the default CPU EP (MLAS) which is fast and fully covers
     * every op in this UNet.                                          */
    /* st = OrtSessionOptionsAppendExecutionProvider_Dnnl(so, 1); */

    st = g_ort->CreateSession(g_env, model_path, so, &g_session);
    g_ort->ReleaseSessionOptions(so);
    CHECK_STATUS(st, "CreateSession");

    st = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                    &g_meminfo);
    CHECK_STATUS(st, "CreateCpuMemoryInfo");

    return 0;
}

/* Process-static scratch buffers for the Fortran-column-major ↔ C-row-
 * major transpose around the ORT Run() call.  Sized for the b16 model's
 * (1, 1, 24, 40) input/output = 960 floats; we allow up to 4 096 for
 * safety against larger models.                                        */
#define ORT_SCRATCH_MAX 4096
static float g_in_c [ORT_SCRATCH_MAX];
static float g_out_c[ORT_SCRATCH_MAX];

int ort_forward_4d_f32(const float *input,
                       float       *output,
                       int          batch,
                       int          channels,
                       int          height,
                       int          width)
{
    if (!g_session) {
        fprintf(stderr, "[ort_wrapper] ort_forward called before ort_init\n");
        return 1;
    }

    const int    nelem  = batch * channels * height * width;
    const size_t nbytes = (size_t)nelem * sizeof(float);
    if (nelem > ORT_SCRATCH_MAX) {
        fprintf(stderr, "[ort_wrapper] tensor too large for scratch buffer "
                "(%d > %d)\n", nelem, ORT_SCRATCH_MAX);
        return 1;
    }

    OrtStatus *st;

    /* Lazily create the persistent input/output OrtValues bound to the
     * scratch buffers.  CreateTensorWithDataAsOrtValue is ~10–20 µs;
     * doing it once vs once-per-call saves ~10 s over an hour of WW3   */
    if (!g_in_val || height != g_dims_h || width != g_dims_w) {
        if (g_in_val)  { g_ort->ReleaseValue(g_in_val);  g_in_val  = NULL; }
        if (g_out_val) { g_ort->ReleaseValue(g_out_val); g_out_val = NULL; }
        const int64_t dims[4] = { (int64_t)batch, (int64_t)channels,
                                  (int64_t)height, (int64_t)width };
        st = g_ort->CreateTensorWithDataAsOrtValue(
            g_meminfo, (void *)g_in_c, nbytes,
            dims, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &g_in_val);
        CHECK_STATUS(st, "CreateTensorWithDataAsOrtValue(in, persistent)");
        st = g_ort->CreateTensorWithDataAsOrtValue(
            g_meminfo, (void *)g_out_c, nbytes,
            dims, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &g_out_val);
        CHECK_STATUS(st, "CreateTensorWithDataAsOrtValue(out, persistent)");
        g_dims_h = height;
        g_dims_w = width;
    }

    /* Fortran is column-major: element (h, w) at index h + w*height.
     * ORT expects C row-major: element (h, w) at index h*width + w.   */
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            g_in_c[h*width + w] = input[h + w*height];
        }
    }

    const char *in_names[1]  = { kInputName  };
    const char *out_names[1] = { kOutputName };
    const OrtValue *in_vals[1] = { g_in_val  };
    OrtValue       *out_vals[1] = { g_out_val };

    st = g_ort->Run(g_session, /*run_options*/ NULL,
                    in_names,  in_vals,  1,
                    out_names, 1, out_vals);
    CHECK_STATUS(st, "Run");
    g_n_calls++;

    /* Transpose output C-row-major -> Fortran column-major.            */
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            output[h + w*height] = g_out_c[h*width + w];
        }
    }
    return 0;
}

void ort_finalize(void)
{
    /* report per-rank call count to stderr (rank id read from MPI env) */
    const char *rk = getenv("OMPI_COMM_WORLD_RANK");
    if (!rk) rk = getenv("PMI_RANK");
    if (!rk) rk = "?";
    fprintf(stderr, "[ort_wrapper] rank %s: ort_forward calls = %lld\n",
            rk, g_n_calls);
    if (g_in_val)  { g_ort->ReleaseValue(g_in_val);       g_in_val  = NULL; }
    if (g_out_val) { g_ort->ReleaseValue(g_out_val);      g_out_val = NULL; }
    if (g_meminfo) { g_ort->ReleaseMemoryInfo(g_meminfo); g_meminfo = NULL; }
    if (g_session) { g_ort->ReleaseSession(g_session);    g_session = NULL; }
    if (g_env)     { g_ort->ReleaseEnv(g_env);            g_env     = NULL; }
}
