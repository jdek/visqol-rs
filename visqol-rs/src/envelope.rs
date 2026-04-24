use crate::fast_fourier_transform;
use crate::fft_manager::FftManager;
use crate::math_utils;
use ndarray::Array1;
use num::complex::Complex64;
use num::Zero;

/// Calculates the upper envelope for a given time domain signal.
///
/// Computes the same value the upstream FFT-based path produces, without
/// running any FFT. The original implementation
/// (`calculate_hilbert` → `inverse_1d` → `* 2.0 - 0.000001` → `.norm()`)
/// passes the analytic-signal IFFT through the helper
/// `fast_fourier_transform::inverse_1d`, which **discards the imaginary
/// component** (see the `"This makes very little sense but oh well…"`
/// comment in that function). What's left is just the real part of
/// `IFFT(X · v_hilbert)` which, for the visqol scaling vector
/// `v = [1, 2, …, 2, v_Nyq, 0, …, 0]`, equals
/// `signal_centered_padded[n] − (X[N/2] / N)·(−1)^n` in the typical case
/// (signal.len() < fft_size, so v_Nyq = 0). Multiplied by 2, shifted by
/// −0.000001, normed and offset by the mean, that's a closed-form O(L)
/// expression — the only piece we need from the spectrum is the Nyquist
/// coefficient `X[N/2]`, which is itself just `Σ pad_centered[k]·(−1)^k`.
pub fn calculate_upper_env(signal: &Array1<f64>) -> Option<ndarray::Array1<f64>> {
    let mean = signal.mean()?;
    let signal_slice = signal.as_slice()?;
    let l = signal_slice.len();

    // Output length: the original `inverse_1d` returned a Vec of length
    // `samples_per_channel == signal.len()`, so envelope length == L.
    let mut env = Array1::<f64>::zeros(l);
    if l == 0 {
        return Some(env);
    }

    // Compute fft_size = next_pow_two(L).max(MIN_FFT_SIZE) — must match
    // FftManager's sizing rule so that X[N/2] / N divides correctly.
    const MIN_FFT_SIZE: usize = 32;
    let n = math_utils::next_pow_two(l).max(MIN_FFT_SIZE);

    // Nyquist coefficient of the centered+zero-padded signal:
    //   X[N/2] = Σ_{k=0..L-1} (signal[k] - mean) · (-1)^k
    // Pad region contributes 0.
    let mut nyquist = 0.0f64;
    let mut sign = 1.0f64;
    for &s in signal_slice {
        nyquist += sign * (s - mean);
        sign = -sign;
    }
    let nyquist_over_n = nyquist / n as f64;

    // Assemble envelope. The original drops the imaginary part of the
    // analytic signal, so this reduces to the absolute value of a scaled
    // mean-centered signal with a small Nyquist-aliasing correction.
    let eps = 0.000_001;
    let mut pos_sign = true; // (-1)^i with i starting at 0
    for (i, e) in env.iter_mut().enumerate() {
        let pad_centered = signal_slice[i] - mean;
        let nyq_term = if pos_sign { nyquist_over_n } else { -nyquist_over_n };
        let re = 2.0 * (pad_centered - nyq_term) - eps;
        // |re| (im was zeroed by the upstream inverse_1d quirk).
        *e = re.abs() + mean;
        pos_sign = !pos_sign;
        let _ = i; // silence unused-var if compiler complains
    }
    Some(env)
}

/// Calculates the hilbert transform for a given time domain signal.
pub fn calculate_hilbert(signal: &mut [f64]) -> Option<Array1<Complex64>> {
    calculate_hilbert_with_offset(signal, 0.0)
}

fn calculate_hilbert_with_offset(signal: &[f64], offset: f64) -> Option<Array1<Complex64>> {
    FftManager::with_cached(signal.len(), |fft_manager| {
        let mut freq_domain_signal = vec![Complex64::zero(); fft_manager.fft_size];
        // Stream `signal[i] + offset` through the FFT input copy.
        fft_manager.freq_from_time_domain_iter(
            signal.iter().map(|&s| s + offset),
            &mut freq_domain_signal,
        );

        let is_odd = signal.len() % 2 == 1;
        let is_non_empty = !signal.is_empty();

        // Apply the Hilbert scaling directly into freq_domain_signal in place.
        // Matches the prior semantics: build the conceptual scaling vector
        //   v[0] = 1
        //   v[signal.len()/2] = 1 (even input) or 2 (odd input)  -- pre-fill
        //   v[1..n]            = 2                                -- final overwrite
        //   v[n..]             = 0
        // where n = freq_domain_signal.len()/2 for even-len fft (always, since
        // fft_size is a power of 2 ≥ 32). The explicit pre-fill only survives
        // for the indices outside [1..n).
        let len = freq_domain_signal.len();
        let n = if is_odd { len.div_ceil(2) } else { len / 2 };
        let half_sig = signal.len() / 2;

        // Capture special-index original value before zeroing the tail.
        let nyquist_scale = if is_non_empty && half_sig >= n {
            Some(if is_odd { 2.0 } else { 1.0 })
        } else {
            None
        };
        let nyquist_orig = nyquist_scale.map(|s| {
            let c = freq_domain_signal[half_sig];
            (s, c)
        });

        // Zero everything from n onward (these indices are 0 in the scale vec
        // unless half_sig falls in this range, which we restore below).
        for c in &mut freq_domain_signal[n..] {
            c.re = 0.0;
            c.im = 0.0;
        }
        // Apply ×2 to indices [1..n).
        for c in &mut freq_domain_signal[1..n] {
            c.re *= 2.0;
            c.im *= 2.0;
        }
        // Restore explicit half-signal index (only meaningful when ≥ n).
        if let Some((scale, orig)) = nyquist_orig {
            freq_domain_signal[half_sig].re = orig.re * scale;
            freq_domain_signal[half_sig].im = orig.im * scale;
        }

        let mut hilbert =
            fast_fourier_transform::inverse_1d(fft_manager, freq_domain_signal.as_slice());
        hilbert
            .iter_mut()
            .for_each(|element| *element = *element * 2.0 - 0.000001);
        Some(Array1::<Complex64>::from_vec(hilbert))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        audio_signal::AudioSignal,
        audio_utils::load_as_mono,
        fft_manager,
        xcorr::{
            calculate_best_lag, calculate_fft_pointwise_product,
            calculate_inverse_fft_pointwise_product, frexp,
        },
    };
    use approx::assert_abs_diff_eq;

    #[test]
    fn hilbert_transform_on_audio_signal() {
        let (mut signal, _) = load_audio_files();
        let result = calculate_hilbert(signal.data_matrix.as_slice_mut().unwrap()).unwrap();

        assert_abs_diff_eq!(result[0].re, 0.000_303_661_691_188_833, epsilon = 0.0001);
    }

    #[test]
    fn envelope_on_audio_signal() {
        let (signal, _) = load_audio_files();
        let result = calculate_upper_env(&signal.data_matrix).unwrap();

        assert_abs_diff_eq!(result[0], 0.00030159861338215923, epsilon = 0.0001);
    }

    #[test]
    fn xcorr_pointwise_prod_on_audio_signal() {
        let (ref_signal, deg_signal) = load_audio_files();
        let ref_signal_vec = ref_signal.data_matrix.to_vec();

        let (_, exponent) = frexp((ref_signal_vec.len() * 2 - 1) as f64);
        let fft_points = 2i32.pow(exponent as u32) as usize;
        let mut manager = fft_manager::FftManager::new(fft_points);

        let result = calculate_fft_pointwise_product(
            &ref_signal.data_matrix.to_vec(),
            &deg_signal.data_matrix.to_vec(),
            &mut manager,
            fft_points,
        );

        assert_abs_diff_eq!(result[0].re, 0.012231532484292984, epsilon = 0.001);
    }

    #[test]
    fn calculate_inverse_fft_pointwise_product_on_audio_pair() {
        let (ref_signal, deg_signal) = load_audio_files();

        let result = calculate_inverse_fft_pointwise_product(
            &mut ref_signal.data_matrix.to_vec(),
            &mut deg_signal.data_matrix.to_vec(),
        );

        assert_abs_diff_eq!(result[0], 79.66060597338944, epsilon = 0.0001);
    }

    #[test]
    fn calculate_best_lag_on_audio_signal() {
        let (ref_signal, deg_signal) = load_audio_files();

        let result = calculate_best_lag(
            ref_signal.data_matrix.as_slice().unwrap(),
            deg_signal.data_matrix.as_slice().unwrap(),
        )
        .unwrap();

        assert_abs_diff_eq!(result, 0);
    }

    fn load_audio_files() -> (AudioSignal, AudioSignal) {
        let ref_signal_path = "test_data/clean_speech/CA01_01.wav";
        let deg_signal_path = "test_data/clean_speech/transcoded_CA01_01.wav";
        (
            load_as_mono(ref_signal_path).unwrap(),
            load_as_mono(deg_signal_path).unwrap(),
        )
    }
}
