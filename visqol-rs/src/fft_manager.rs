use crate::math_utils;
use num::complex::Complex64;
use num::Zero;
use rustfft::FftPlanner;
use std::cell::RefCell;

// Constants
const MIN_FFT_SIZE: usize = 32;

// Thread-local FFT planner to cache plans across FftManager instances.
// RustFFT's FftPlanner caches internally, so sharing one avoids re-planning.
thread_local! {
    static PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
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
        PLANNER.with(|p| {
            let fft = p.borrow_mut().plan_fft_forward(self.fft_size);
            if time_channel.len() != self.fft_size {
                time_channel.resize(self.fft_size, 0.0f64);
            }
            // Write real values into pre-allocated complex buffer (avoids allocation)
            for (c, &r) in self.complex_buf.iter_mut().zip(time_channel.iter()) {
                c.re = r;
                c.im = 0.0;
            }
            self.ensure_scratch(fft.get_outofplace_scratch_len());
            fft.process_outofplace_with_scratch(
                &mut self.complex_buf[..],
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
}
