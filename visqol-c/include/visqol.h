#ifndef VISQOL_H
#define VISQOL_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Opaque handle to a ViSQOL instance.
 */
typedef struct VisqolHandle VisqolHandle;

/**
 * Opaque handle to a prepared reference signal.
 */
typedef struct VisqolRef VisqolRef;

/**
 * Result of a ViSQOL comparison.
 */
typedef struct VisqolResult {
  double moslqo;
  double vnsim;
} VisqolResult;

/**
 * Create a wideband (speech) ViSQOL instance.
 *
 * `use_unscaled_mos_mapping`: if true, perfect NSIM scores result in ~4.x instead of 5.0.
 * `search_window_radius`: how far to search for patch matches (default: 60).
 *
 * Returns a handle that must be freed with `visqol_destroy`.
 */
struct VisqolHandle *visqol_create_wideband(bool use_unscaled_mos_mapping,
                                            uintptr_t search_window_radius);

/**
 * Create a fullband (audio) ViSQOL instance using the embedded default SVM model.
 *
 * `search_window_radius`: how far to search for patch matches (default: 60).
 *
 * Returns a handle that must be freed with `visqol_destroy`.
 */
struct VisqolHandle *visqol_create_fullband(uintptr_t search_window_radius);

/**
 * Create a fullband (audio) ViSQOL instance with a custom SVM model loaded from a file path.
 *
 * `model_path`: null-terminated path to a libSVM model file.
 * `search_window_radius`: how far to search for patch matches (default: 60).
 *
 * Returns a handle that must be freed with `visqol_destroy`.
 * Returns null if `model_path` is null or not valid UTF-8.
 */
struct VisqolHandle *visqol_create_fullband_with_model(const char *model_path,
                                                       uintptr_t search_window_radius);

/**
 * Run a comparison between a reference and degraded audio file.
 *
 * `handle`: a ViSQOL instance created with `visqol_create_*`.
 * `reference_path`: null-terminated path to the reference WAV file.
 * `degraded_path`: null-terminated path to the degraded WAV file.
 * `result_out`: pointer to a `VisqolResult` struct that will be filled on success.
 *
 * Returns 0 on success, -1 on error.
 */
int32_t visqol_run(struct VisqolHandle *handle,
                   const char *reference_path,
                   const char *degraded_path,
                   struct VisqolResult *result_out);

/**
 * Prepare a reference file for repeated comparisons.
 *
 * Pre-computes the reference spectrogram and patch indices so they can
 * be reused across many degraded files, avoiding redundant work.
 *
 * Returns a handle that must be freed with `visqol_prepared_ref_destroy`.
 * Returns null on error.
 */
struct VisqolRef *visqol_prepare_ref(struct VisqolHandle *handle, const char *reference_path);

/**
 * Run a comparison using a prepared reference.
 *
 * `handle`: a ViSQOL instance.
 * `prepared_ref`: a prepared reference from `visqol_prepare_ref`.
 * `degraded_path`: null-terminated path to the degraded WAV file.
 * `result_out`: pointer to a `VisqolResult` struct that will be filled on success.
 *
 * Returns 0 on success, -1 on error.
 */
int32_t visqol_run_with_ref(struct VisqolHandle *handle,
                            const struct VisqolRef *prepared_ref,
                            const char *degraded_path,
                            struct VisqolResult *result_out);

/**
 * Destroy a prepared reference and free its resources.
 *
 * Passing null is a no-op.
 */
void visqol_prepared_ref_destroy(struct VisqolRef *prepared_ref);

/**
 * Run a comparison from raw PCM buffers (no file I/O).
 *
 * Both `ref_pcm` and `deg_pcm` are mono i16 PCM at `sample_rate` Hz.
 * `ref_len` and `deg_len` are the number of samples (not bytes).
 *
 * Returns 0 on success, -1 on error.
 */
int32_t visqol_run_pcm(struct VisqolHandle *handle,
                       const int16_t *ref_pcm,
                       uintptr_t ref_len,
                       const int16_t *deg_pcm,
                       uintptr_t deg_len,
                       uint32_t sample_rate,
                       struct VisqolResult *result_out);

/**
 * Destroy a ViSQOL instance and free its resources.
 *
 * Passing null is a no-op.
 */
void visqol_destroy(struct VisqolHandle *handle);

#endif  /* VISQOL_H */
