/* ort_wrapper.h
 *
 * Minimal Fortran-friendly C ABI around onnxruntime_c_api.h.  Exposes just
 * what w3snl6md.F90 needs:
 *
 *   - ort_init               : process-wide one-shot (creates Env + Session)
 *   - ort_forward_4d_f32     : one-input / one-output FP32 4-D forward
 *   - ort_finalize           : release Session + Env
 *
 * The session is process-global (a single SAVE'd handle in Fortran),
 * matching how the current FTorch-based code uses ML_MODEL.
 *
 * Threading: the underlying ONNX Runtime InferenceSession is thread-safe
 * for Run() calls.  We configure intra_op_num_threads=1 so each MPI rank
 * doesn't spawn parasitic workers (mirroring the current libtorch single-
 * thread-per-rank setup at np=8).
 *
 * Returns 0 on success, non-zero on error (the C side prints the ORT
 * error string to stderr before returning).
 */

#ifndef ORT_WRAPPER_H
#define ORT_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Load the .onnx model from disk and prepare a single InferenceSession.
 * model_path must be NUL-terminated.  Safe to call once per process.       */
int ort_init(const char *model_path);

/* Run forward on a single FP32 4-D input tensor of shape
 *   [batch, channels, height, width].
 * input and output must both be contiguous arrays sized accordingly.
 * Output buffer is overwritten in place.                                   */
int ort_forward_4d_f32(const float *input,
                       float       *output,
                       int          batch,
                       int          channels,
                       int          height,
                       int          width);

/* Release the InferenceSession and the global Env.  Idempotent.            */
void ort_finalize(void);

#ifdef __cplusplus
}
#endif

#endif /* ORT_WRAPPER_H */
