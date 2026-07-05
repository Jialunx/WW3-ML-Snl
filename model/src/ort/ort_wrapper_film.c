/* ort_wrapper_film.c -- 2-input ONNX Runtime wrapper for ML-FiLM.
 *
 * Owns one process-wide InferenceSession.  Two FP32 inputs:
 *   efth  : 4-D [batch, channels, height, width]  (the spectrum)
 *   depth : 2-D [batch, 1]                         (the scalar depth)
 * One FP32 4-D output:
 *   snl   : same shape as efth.
 *
 * The exported FiLM ONNX (cond_unet_film_24x40.onnx) wraps the entire
 * recipe in-graph, so the Fortran side only fills efth + depth and rescales
 * the returned S by conx = cg/(2*pi*sigma).  Node names from the export:
 *   torch.onnx.export(..., input_names=['efth','depth'], output_names=['snl'])
 */

#include "ort_wrapper_film.h"
#include <onnxruntime_c_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *kEfthName  = "efth";
static const char *kDepthName = "depth";
static const char *kOutName   = "snl";

static const OrtApi  *g_ort     = NULL;
static OrtEnv        *g_env     = NULL;
static OrtSession    *g_session = NULL;
static OrtMemoryInfo *g_meminfo = NULL;
static OrtValue      *g_efth_val = NULL;
static OrtValue      *g_dep_val  = NULL;
static OrtValue      *g_out_val  = NULL;
static int            g_dims_h   = 0;
static int            g_dims_w   = 0;
static long long      g_n_calls  = 0;

#define CHECK_STATUS(status, where)                                     \
    do {                                                                \
        if (status != NULL) {                                           \
            const char *msg = g_ort->GetErrorMessage(status);           \
            fprintf(stderr, "[ort_wrapper_film] %s: %s\n", where, msg); \
            g_ort->ReleaseStatus(status);                               \
            return 1;                                                   \
        }                                                               \
    } while (0)

static void ort_dump_counts_atexit(void)
{
    const char *rk = getenv("OMPI_COMM_WORLD_RANK");
    if (!rk) rk = getenv("PMI_RANK");
    if (!rk) rk = "?";
    fprintf(stderr, "[ort_wrapper_film] rank %s: ort_forward calls = %lld\n",
            rk, g_n_calls);
    fflush(stderr);
}

int ort_init(const char *model_path)
{
    if (g_session) return 0;  /* idempotent */
    atexit(ort_dump_counts_atexit);

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "[ort_wrapper_film] GetApi() returned NULL\n");
        return 1;
    }

    OrtStatus *st;
    st = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ww3_ort_film", &g_env);
    CHECK_STATUS(st, "CreateEnv");

    OrtSessionOptions *so;
    st = g_ort->CreateSessionOptions(&so);
    CHECK_STATUS(st, "CreateSessionOptions");
    st = g_ort->SetIntraOpNumThreads(so, 1);
    CHECK_STATUS(st, "SetIntraOpNumThreads");
    st = g_ort->SetInterOpNumThreads(so, 1);
    CHECK_STATUS(st, "SetInterOpNumThreads");
    st = g_ort->SetSessionExecutionMode(so, ORT_SEQUENTIAL);
    CHECK_STATUS(st, "SetSessionExecutionMode");
    st = g_ort->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL);
    CHECK_STATUS(st, "SetSessionGraphOptimizationLevel");

    st = g_ort->CreateSession(g_env, model_path, so, &g_session);
    g_ort->ReleaseSessionOptions(so);
    CHECK_STATUS(st, "CreateSession");

    st = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                    &g_meminfo);
    CHECK_STATUS(st, "CreateCpuMemoryInfo");
    return 0;
}

#define ORT_SCRATCH_MAX 4096
static float g_in_c [ORT_SCRATCH_MAX];
static float g_out_c[ORT_SCRATCH_MAX];
static float g_dep_c[4];

int ort_forward_film(const float *efth, const float *depth, float *output,
                     int batch, int channels, int height, int width)
{
    if (!g_session) {
        fprintf(stderr, "[ort_wrapper_film] forward called before init\n");
        return 1;
    }
    const int    nelem  = batch * channels * height * width;
    const size_t nbytes = (size_t)nelem * sizeof(float);
    if (nelem > ORT_SCRATCH_MAX) {
        fprintf(stderr, "[ort_wrapper_film] tensor too large (%d > %d)\n",
                nelem, ORT_SCRATCH_MAX);
        return 1;
    }

    OrtStatus *st;

    if (!g_efth_val || height != g_dims_h || width != g_dims_w) {
        if (g_efth_val) { g_ort->ReleaseValue(g_efth_val); g_efth_val = NULL; }
        if (g_dep_val)  { g_ort->ReleaseValue(g_dep_val);  g_dep_val  = NULL; }
        if (g_out_val)  { g_ort->ReleaseValue(g_out_val);  g_out_val  = NULL; }
        const int64_t dims[4] = { (int64_t)batch, (int64_t)channels,
                                  (int64_t)height, (int64_t)width };
        const int64_t ddims[2] = { (int64_t)batch, 1 };
        st = g_ort->CreateTensorWithDataAsOrtValue(
            g_meminfo, (void *)g_in_c, nbytes, dims, 4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &g_efth_val);
        CHECK_STATUS(st, "CreateTensor(efth)");
        st = g_ort->CreateTensorWithDataAsOrtValue(
            g_meminfo, (void *)g_dep_c, (size_t)batch * sizeof(float),
            ddims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &g_dep_val);
        CHECK_STATUS(st, "CreateTensor(depth)");
        st = g_ort->CreateTensorWithDataAsOrtValue(
            g_meminfo, (void *)g_out_c, nbytes, dims, 4,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &g_out_val);
        CHECK_STATUS(st, "CreateTensor(out)");
        g_dims_h = height;
        g_dims_w = width;
    }

    /* Fortran column-major (h + w*height) -> C row-major (h*width + w). */
    for (int h = 0; h < height; ++h)
        for (int w = 0; w < width; ++w)
            g_in_c[h*width + w] = efth[h + w*height];
    for (int b = 0; b < batch; ++b)
        g_dep_c[b] = depth[b];

    const char     *in_names[2] = { kEfthName, kDepthName };
    const OrtValue *in_vals[2]  = { g_efth_val, g_dep_val };
    const char     *out_names[1] = { kOutName };
    OrtValue       *out_vals[1]  = { g_out_val };

    st = g_ort->Run(g_session, NULL, in_names, in_vals, 2,
                    out_names, 1, out_vals);
    CHECK_STATUS(st, "Run");
    g_n_calls++;

    for (int h = 0; h < height; ++h)
        for (int w = 0; w < width; ++w)
            output[h + w*height] = g_out_c[h*width + w];
    return 0;
}

void ort_finalize(void)
{
    const char *rk = getenv("OMPI_COMM_WORLD_RANK");
    if (!rk) rk = getenv("PMI_RANK");
    if (!rk) rk = "?";
    fprintf(stderr, "[ort_wrapper_film] rank %s: ort_forward calls = %lld\n",
            rk, g_n_calls);
    if (g_efth_val) { g_ort->ReleaseValue(g_efth_val);    g_efth_val = NULL; }
    if (g_dep_val)  { g_ort->ReleaseValue(g_dep_val);     g_dep_val  = NULL; }
    if (g_out_val)  { g_ort->ReleaseValue(g_out_val);     g_out_val  = NULL; }
    if (g_meminfo)  { g_ort->ReleaseMemoryInfo(g_meminfo); g_meminfo = NULL; }
    if (g_session)  { g_ort->ReleaseSession(g_session);   g_session  = NULL; }
    if (g_env)      { g_ort->ReleaseEnv(g_env);           g_env      = NULL; }
}
