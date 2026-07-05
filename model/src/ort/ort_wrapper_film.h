/* ort_wrapper_film.h -- 2-input (efth, depth) ONNX Runtime wrapper for the
 * depth-conditional ML-FiLM S_nl operator.  Mirrors ort_wrapper.h but the
 * forward takes two inputs (the spectrum and the scalar depth) because the
 * whole FiLM recipe is in-graph; the ONNX node names are efth/depth/snl. */
#ifndef ORT_WRAPPER_FILM_H
#define ORT_WRAPPER_FILM_H
#ifdef __cplusplus
extern "C" {
#endif
int  ort_init(const char *model_path);
int  ort_forward_film(const float *efth, const float *depth, float *output,
                      int batch, int channels, int height, int width);
void ort_finalize(void);
#ifdef __cplusplus
}
#endif
#endif /* ORT_WRAPPER_FILM_H */
