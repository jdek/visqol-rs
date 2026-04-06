use std::ffi::CStr;
use std::os::raw::c_char;

use visqol_rs::audio_signal::AudioSignal;
use visqol_rs::constants::{NUM_BANDS_AUDIO, NUM_BANDS_SPEECH};
use visqol_rs::support_vector_regression_model::SupportVectorRegressionModel;
use visqol_rs::variant::Variant;
use visqol_rs::visqol_manager::VisqolManager;
use visqol_rs::VisqolRef;

const DEFAULT_MODEL_CONTENT: &str = include_str!("../../model/libsvm_nu_svr_model.txt");

enum Manager {
    Speech(VisqolManager<NUM_BANDS_SPEECH>),
    Audio(VisqolManager<NUM_BANDS_AUDIO>),
}

/// Opaque handle to a ViSQOL instance.
pub struct VisqolHandle {
    manager: Manager,
}

/// Result of a ViSQOL comparison.
#[repr(C)]
pub struct VisqolResult {
    pub moslqo: f64,
    pub vnsim: f64,
}

/// Create a wideband (speech) ViSQOL instance.
///
/// `use_unscaled_mos_mapping`: if true, perfect NSIM scores result in ~4.x instead of 5.0.
/// `search_window_radius`: how far to search for patch matches (default: 60).
///
/// Returns a handle that must be freed with `visqol_destroy`.
#[no_mangle]
pub extern "C" fn visqol_create_wideband(
    use_unscaled_mos_mapping: bool,
    search_window_radius: usize,
) -> *mut VisqolHandle {
    let variant = Variant::Wideband {
        use_unscaled_mos_mapping,
    };
    let manager = VisqolManager::new(variant, search_window_radius);
    Box::into_raw(Box::new(VisqolHandle {
        manager: Manager::Speech(manager),
    }))
}

/// Create a fullband (audio) ViSQOL instance using the embedded default SVM model.
///
/// `search_window_radius`: how far to search for patch matches (default: 60).
///
/// Returns a handle that must be freed with `visqol_destroy`.
#[no_mangle]
pub extern "C" fn visqol_create_fullband(search_window_radius: usize) -> *mut VisqolHandle {
    let model = SupportVectorRegressionModel::from_model_content(DEFAULT_MODEL_CONTENT);
    let variant = Variant::Fullband { model };
    let manager = VisqolManager::new(variant, search_window_radius);
    Box::into_raw(Box::new(VisqolHandle {
        manager: Manager::Audio(manager),
    }))
}

/// Create a fullband (audio) ViSQOL instance with a custom SVM model loaded from a file path.
///
/// `model_path`: null-terminated path to a libSVM model file.
/// `search_window_radius`: how far to search for patch matches (default: 60).
///
/// Returns a handle that must be freed with `visqol_destroy`.
/// Returns null if `model_path` is null or not valid UTF-8.
#[no_mangle]
pub extern "C" fn visqol_create_fullband_with_model(
    model_path: *const c_char,
    search_window_radius: usize,
) -> *mut VisqolHandle {
    if model_path.is_null() {
        return std::ptr::null_mut();
    }
    let path = match unsafe { CStr::from_ptr(model_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let model = SupportVectorRegressionModel::new(path);
    let variant = Variant::Fullband { model };
    let manager = VisqolManager::new(variant, search_window_radius);
    Box::into_raw(Box::new(VisqolHandle {
        manager: Manager::Audio(manager),
    }))
}

/// Run a comparison between a reference and degraded audio file.
///
/// `handle`: a ViSQOL instance created with `visqol_create_*`.
/// `reference_path`: null-terminated path to the reference WAV file.
/// `degraded_path`: null-terminated path to the degraded WAV file.
/// `result_out`: pointer to a `VisqolResult` struct that will be filled on success.
///
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn visqol_run(
    handle: *mut VisqolHandle,
    reference_path: *const c_char,
    degraded_path: *const c_char,
    result_out: *mut VisqolResult,
) -> i32 {
    if handle.is_null() || reference_path.is_null() || degraded_path.is_null() || result_out.is_null() {
        return -1;
    }

    let h = unsafe { &mut *handle };
    let ref_path = match unsafe { CStr::from_ptr(reference_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let deg_path = match unsafe { CStr::from_ptr(degraded_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let result = match &mut h.manager {
        Manager::Speech(m) => m.run(ref_path, deg_path),
        Manager::Audio(m) => m.run(ref_path, deg_path),
    };

    match result {
        Ok(sim) => {
            unsafe {
                (*result_out).moslqo = sim.moslqo;
                (*result_out).vnsim = sim.vnsim;
            }
            0
        }
        Err(_) => -1,
    }
}

/// Prepare a reference file for repeated comparisons.
///
/// Pre-computes the reference spectrogram and patch indices so they can
/// be reused across many degraded files, avoiding redundant work.
///
/// Returns a handle that must be freed with `visqol_prepared_ref_destroy`.
/// Returns null on error.
#[no_mangle]
pub extern "C" fn visqol_prepare_ref(
    handle: *mut VisqolHandle,
    reference_path: *const c_char,
) -> *mut VisqolRef {
    if handle.is_null() || reference_path.is_null() {
        return std::ptr::null_mut();
    }
    let h = unsafe { &mut *handle };
    let ref_path = match unsafe { CStr::from_ptr(reference_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let result = match &mut h.manager {
        Manager::Speech(m) => m.prepare_reference(ref_path),
        Manager::Audio(m) => m.prepare_reference(ref_path),
    };

    match result {
        Ok(prepared) => Box::into_raw(Box::new(prepared)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Run a comparison using a prepared reference.
///
/// `handle`: a ViSQOL instance.
/// `prepared_ref`: a prepared reference from `visqol_prepare_ref`.
/// `degraded_path`: null-terminated path to the degraded WAV file.
/// `result_out`: pointer to a `VisqolResult` struct that will be filled on success.
///
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn visqol_run_with_ref(
    handle: *mut VisqolHandle,
    prepared_ref: *const VisqolRef,
    degraded_path: *const c_char,
    result_out: *mut VisqolResult,
) -> i32 {
    if handle.is_null() || prepared_ref.is_null() || degraded_path.is_null() || result_out.is_null()
    {
        return -1;
    }

    let h = unsafe { &mut *handle };
    let prepared = unsafe { &*prepared_ref };
    let deg_path = match unsafe { CStr::from_ptr(degraded_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let result = match &mut h.manager {
        Manager::Speech(m) => m.run_with_reference(prepared, deg_path),
        Manager::Audio(m) => m.run_with_reference(prepared, deg_path),
    };

    match result {
        Ok(sim) => {
            unsafe {
                (*result_out).moslqo = sim.moslqo;
                (*result_out).vnsim = sim.vnsim;
            }
            0
        }
        Err(_) => -1,
    }
}

/// Destroy a prepared reference and free its resources.
///
/// Passing null is a no-op.
#[no_mangle]
pub extern "C" fn visqol_prepared_ref_destroy(prepared_ref: *mut VisqolRef) {
    if !prepared_ref.is_null() {
        unsafe {
            drop(Box::from_raw(prepared_ref));
        }
    }
}

/// Run a comparison from raw PCM buffers (no file I/O).
///
/// Both `ref_pcm` and `deg_pcm` are mono i16 PCM at `sample_rate` Hz.
/// `ref_len` and `deg_len` are the number of samples (not bytes).
///
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn visqol_run_pcm(
    handle: *mut VisqolHandle,
    ref_pcm: *const i16,
    ref_len: usize,
    deg_pcm: *const i16,
    deg_len: usize,
    sample_rate: u32,
    result_out: *mut VisqolResult,
) -> i32 {
    if handle.is_null() || ref_pcm.is_null() || deg_pcm.is_null() || result_out.is_null() {
        return -1;
    }

    let ref_slice = unsafe { std::slice::from_raw_parts(ref_pcm, ref_len) };
    let deg_slice = unsafe { std::slice::from_raw_parts(deg_pcm, deg_len) };

    let ref_f64: Vec<f64> = ref_slice.iter().map(|&s| s as f64 / 32768.0).collect();
    let deg_f64: Vec<f64> = deg_slice.iter().map(|&s| s as f64 / 32768.0).collect();

    let mut ref_signal = AudioSignal::new(&ref_f64, sample_rate);
    let mut deg_signal = AudioSignal::new(&deg_f64, sample_rate);

    let h = unsafe { &mut *handle };
    let result = match &mut h.manager {
        Manager::Speech(m) => m.compute_results(&mut ref_signal, &mut deg_signal),
        Manager::Audio(m) => m.compute_results(&mut ref_signal, &mut deg_signal),
    };

    match result {
        Ok(sim) => {
            unsafe {
                (*result_out).moslqo = sim.moslqo;
                (*result_out).vnsim = sim.vnsim;
            }
            0
        }
        Err(_) => -1,
    }
}

/// Destroy a ViSQOL instance and free its resources.
///
/// Passing null is a no-op.
#[no_mangle]
pub extern "C" fn visqol_destroy(handle: *mut VisqolHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}
