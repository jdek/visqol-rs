use crate::math_utils;
use num::complex::Complex64;
use num::Zero;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::FftPlanner;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

// Constants
const MIN_FFT_SIZE: usize = 32;

// Thread-local FFT planner to cache plans across FftManager instances.
// RustFFT's FftPlanner caches internally, so sharing one avoids re-planning.
thread_local! {
    static PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
    static REAL_PLANNER: RefCell<RealFftPlanner<f64>> = RefCell::new(RealFftPlanner::new());
    // Thread-local cache of FftManager instances keyed by samples_per_channel.
    // Reuses the per-manager complex_buf and scratch_buf across calls in the
    // envelope/xcorr hot path of finely_align_and_recreate_patches.
    static MANAGER_CACHE: RefCell<HashMap<usize, FftManager>> = RefCell::new(HashMap::new());
}

/// Wrapper around the `rustfft` library to perform basic fft operations.
/// Pre-allocates scratch and complex buffers to avoid per-call heap allocation.
pub struct FftManager {
    /// Length of the fft
    pub fft_size: usize,
    /// Scale factor to apply after inverse fft
    inverse_fft_scale: f64,
    /// Number of samples to apply fft on
    pub samples_per_channel: usize,
    /// Reusable complex buffer for real→complex conversion
    complex_buf: Vec<Complex64>,
    /// Reusable scratch buffer for FFT operations
    scratch_buf: Vec<Complex64>,
    /// Reusable real-domain scratch (size = fft_size). Holds R2C input or C2R output.
    real_buf: Vec<f64>,
    /// Reusable complex scratch for the realfft engine.
    real_scratch: Vec<Complex64>,
    /// Cached realfft plans. Plans hold internal mutable scratch, but `process_*_with_scratch`
    /// uses the caller-provided scratch.
    r2c: Option<Arc<dyn RealToComplex<f64>>>,
    c2r: Option<Arc<dyn ComplexToReal<f64>>>,
}

impl FftManager {
    /// Creates a new fft manager, computes internal variables from `samples_per_channel`
    pub fn new(samples_per_channel: usize) -> Self {
        let fft_size = math_utils::next_pow_two(samples_per_channel).max(MIN_FFT_SIZE);

        Self {
            fft_size,
            samples_per_channel,
            inverse_fft_scale: 1.0f64 / (fft_size as f64),
            complex_buf: vec![Complex64::zero(); fft_size],
            scratch_buf: Vec::new(),
            real_buf: Vec::new(),
            real_scratch: Vec::new(),
            r2c: None,
            c2r: None,
        }
    }

    /// Returns the size of the half-spectrum produced by `forward_r2c` /
    /// consumed by `inverse_c2r` for this manager: `fft_size / 2 + 1`.
    pub fn half_spectrum_size(&self) -> usize {
        self.fft_size / 2 + 1
    }

    fn ensure_r2c(&mut self) -> Arc<dyn RealToComplex<f64>> {
        if self.r2c.is_none() {
            let plan = REAL_PLANNER.with(|p| p.borrow_mut().plan_fft_forward(self.fft_size));
            self.r2c = Some(plan);
        }
        self.r2c.as_ref().unwrap().clone()
    }

    fn ensure_c2r(&mut self) -> Arc<dyn ComplexToReal<f64>> {
        if self.c2r.is_none() {
            let plan = REAL_PLANNER.with(|p| p.borrow_mut().plan_fft_inverse(self.fft_size));
            self.c2r = Some(plan);
        }
        self.c2r.as_ref().unwrap().clone()
    }

    /// Real → complex forward FFT. `input` may be shorter than `fft_size`
    /// (zero-padded) or equal length. `output` must have length `half_spectrum_size()`.
    /// The internal `real_buf` is used as the contiguous R2C input scratch.
    pub fn forward_r2c(&mut self, input: &[f64], output: &mut [Complex64]) {
        self.forward_r2c_iter(input.iter().copied(), output);
    }

    /// Same as `forward_r2c` but consumes a real-valued iterator, letting
    /// callers fuse a transform (e.g. mean-centering) into the copy.
    pub fn forward_r2c_iter<I: IntoIterator<Item = f64>>(
        &mut self,
        iter: I,
        output: &mut [Complex64],
    ) {
        debug_assert_eq!(output.len(), self.half_spectrum_size());
        let r2c = self.ensure_r2c();
        let scratch_len = r2c.get_scratch_len();
        if self.real_buf.len() < self.fft_size {
            self.real_buf.resize(self.fft_size, 0.0);
        }
        if self.real_scratch.len() < scratch_len {
            self.real_scratch.resize(scratch_len, Complex64::zero());
        }
        let mut written = 0usize;
        for (slot, v) in self.real_buf[..self.fft_size]
            .iter_mut()
            .zip(iter.into_iter().take(self.fft_size))
        {
            *slot = v;
            written += 1;
        }
        for s in &mut self.real_buf[written..self.fft_size] {
            *s = 0.0;
        }
        r2c.process_with_scratch(
            &mut self.real_buf[..self.fft_size],
            output,
            &mut self.real_scratch[..scratch_len],
        )
        .expect("realfft r2c failed");
    }

    /// Complex (half-spectrum) → real inverse FFT. `input` length must equal
    /// `half_spectrum_size()`. `output` length must equal `fft_size`. The
    /// realfft inverse does NOT scale by `1/N`; this method applies the
    /// `inverse_fft_scale` to match the standard rustfft convention used by
    /// `inverse_1d_conj_sym`. NB: realfft mutates `input`.
    pub fn inverse_c2r(&mut self, input: &mut [Complex64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), self.half_spectrum_size());
        debug_assert_eq!(output.len(), self.fft_size);
        let c2r = self.ensure_c2r();
        let scratch_len = c2r.get_scratch_len();
        if self.real_scratch.len() < scratch_len {
            self.real_scratch.resize(scratch_len, Complex64::zero());
        }
        c2r.process_with_scratch(input, output, &mut self.real_scratch[..scratch_len])
            .expect("realfft c2r failed");
        let s = self.inverse_fft_scale;
        for v in output.iter_mut() {
            *v *= s;
        }
    }

    /// Ensure scratch buffer is large enough for the given FFT plan.
    fn ensure_scratch(&mut self, needed: usize) {
        if self.scratch_buf.len() < needed {
            self.scratch_buf.resize(needed, Complex64::zero());
        }
    }

    /// Zero-pads `time_channel` if necessary, transforms its contents into the frequency domain and stores it in `freq_channel`
    pub fn freq_from_time_domain(
        &mut self,
        time_channel: &mut Vec<f64>,
        freq_channel: &mut [Complex64],
    ) {
        if time_channel.len() != self.fft_size {
            time_channel.resize(self.fft_size, 0.0f64);
        }
        self.freq_from_time_domain_slice(time_channel.as_slice(), freq_channel);
    }

    /// Like `freq_from_time_domain` but accepts a `&[f64]` slice. Zero-pads
    /// into the internal `complex_buf` (avoids the caller-side Vec allocation
    /// + copy of the input signal that the `Vec<f64>` API path forced).
    pub fn freq_from_time_domain_slice(
        &mut self,
        time_channel: &[f64],
        freq_channel: &mut [Complex64],
    ) {
        self.freq_from_time_domain_iter(time_channel.iter().copied(), freq_channel);
    }

    /// Like `freq_from_time_domain_slice` but consumes a real-valued iterator,
    /// letting callers fuse a transform (e.g. mean-centering) into the copy.
    pub fn freq_from_time_domain_iter<I: IntoIterator<Item = f64>>(
        &mut self,
        iter: I,
        freq_channel: &mut [Complex64],
    ) {
        PLANNER.with(|p| {
            let fft = p.borrow_mut().plan_fft_forward(self.fft_size);
            let mut written = 0usize;
            for (c, r) in self.complex_buf[..self.fft_size]
                .iter_mut()
                .zip(iter.into_iter().take(self.fft_size))
            {
                c.re = r;
                c.im = 0.0;
                written += 1;
            }
            for c in &mut self.complex_buf[written..self.fft_size] {
                c.re = 0.0;
                c.im = 0.0;
            }
            self.ensure_scratch(fft.get_outofplace_scratch_len());
            fft.process_outofplace_with_scratch(
                &mut self.complex_buf[..self.fft_size],
                freq_channel,
                &mut self.scratch_buf[..],
            );
        });
    }

    /// Zero-pads `freq_channel` if necessary, transforms its contents into the time domain and stores it in `time_channel`
    pub fn time_from_freq_domain(
        &mut self,
        freq_channel: &mut [Complex64],
        time_channel: &mut Vec<f64>,
    ) {
        PLANNER.with(|p| {
            let fft = p.borrow_mut().plan_fft_inverse(self.fft_size);
            self.ensure_scratch(fft.get_outofplace_scratch_len());
            // Prepare output complex buffer
            let needed = self.fft_size;
            if self.complex_buf.len() < needed {
                self.complex_buf.resize(needed, Complex64::zero());
            }
            fft.process_outofplace_with_scratch(
                freq_channel,
                &mut self.complex_buf[..needed],
                &mut self.scratch_buf[..],
            );
            // Extract real parts directly into time_channel (avoids allocation)
            time_channel.resize(self.fft_size, 0.0);
            for (t, c) in time_channel.iter_mut().zip(self.complex_buf.iter()) {
                *t = c.re;
            }
        });
    }

    /// Multiplies each element in `time_channel` by `self.inverse_fft_scale`
    pub fn apply_reverse_fft_scaling(&self, time_channel: &mut [f64]) {
        time_channel.iter_mut().for_each(|x| {
            *x *= self.inverse_fft_scale;
        });
    }

    /// Run a closure with a thread-local cached FftManager keyed on
    /// `samples_per_channel`. Reuses the per-manager scratch and complex
    /// buffers across calls; the rustfft plan itself is shared via PLANNER.
    pub fn with_cached<R>(samples_per_channel: usize, f: impl FnOnce(&mut FftManager) -> R) -> R {
        MANAGER_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let mgr = cache
                .entry(samples_per_channel)
                .or_insert_with(|| FftManager::new(samples_per_channel));
            f(mgr)
        })
    }
}
