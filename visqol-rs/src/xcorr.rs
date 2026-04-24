use crate::fft_manager::FftManager;
use num::complex::Complex64;
use num::Zero;

/// Calculate the maximum delay between two signals.
pub fn calculate_best_lag(signal_1: &[f64], signal_2: &[f64]) -> Option<i64> {
    let max_lag = ((signal_1.len().max(signal_2.len())) - 1) as i64;

    let mut sig1 = signal_1.to_vec();
    let mut sig2 = signal_2.to_vec();
    let point_wise_fft_vec =
        calculate_inverse_fft_pointwise_product(&mut sig1, &mut sig2);

    // Find best correlation directly without rearranging.
    // The correlation output is laid out as:
    //   [0..max_lag+1]  = positive lags (indices 0..max_lag map to lags 0..max_lag)
    //   [len-max_lag..] = negative lags (index len-k maps to lag -k)
    //
    // We search both regions and track the best absolute correlation.
    let n = point_wise_fft_vec.len();
    let neg_start = n - max_lag as usize;
    let mut best_abs = f64::MIN;
    let mut best_lag_result: i64 = 0;

    // Negative lags region
    for i in neg_start..n {
        let abs_val = point_wise_fft_vec[i].abs();
        if abs_val > best_abs {
            best_abs = abs_val;
            best_lag_result = i as i64 - n as i64; // negative
        }
    }

    // Positive lags region
    for i in 0..=(max_lag as usize) {
        let abs_val = point_wise_fft_vec[i].abs();
        if abs_val > best_abs {
            best_abs = abs_val;
            best_lag_result = i as i64; // positive
        }
    }

    Some(best_lag_result)
}

/// Calculates the pointwise inverse fft product of 2 signals
pub fn calculate_inverse_fft_pointwise_product(
    signal_1: &mut Vec<f64>,
    signal_2: &mut Vec<f64>,
) -> Vec<f64> {
    let biggest_length = signal_1.len().max(signal_2.len());

    match &signal_1.len().cmp(&signal_2.len()) {
        std::cmp::Ordering::Less => {
            signal_1.resize(biggest_length, 0.0);
        }
        std::cmp::Ordering::Greater => {
            signal_2.resize(biggest_length, 0.0);
        }
        _ => {}
    }
    let (_, exp) = frexp((signal_1.len() * 2 - 1) as f64);
    let fft_points = 2usize.pow(exp as u32);
    FftManager::with_cached(fft_points, |manager| {
        // Real → complex forward FFT on each input (inputs are real-valued
        // and zero-padded to fft_points). The pointwise product H1 * conj(H2)
        // is Hermitian-symmetric; an inverse C2R recovers the real-valued
        // cross-correlation. Halves both forward and inverse FFT work
        // compared to the previous full-complex round-trip.
        let half = manager.half_spectrum_size();
        let mut h1 = vec![Complex64::zero(); half];
        let mut h2 = vec![Complex64::zero(); half];
        manager.forward_r2c(signal_1, &mut h1);
        manager.forward_r2c(signal_2, &mut h2);
        // h1[i] *= conj(h2[i])
        for (a, b) in h1.iter_mut().zip(h2.iter()) {
            // (ar + i*ai) * (br - i*bi) = (ar*br + ai*bi) + i*(ai*br - ar*bi)
            let ar = a.re;
            let ai = a.im;
            let br = b.re;
            let bi = b.im;
            a.re = ar * br + ai * bi;
            a.im = ai * br - ar * bi;
        }
        let mut out = vec![0.0f64; manager.fft_size];
        manager.inverse_c2r(&mut h1, &mut out);
        out.truncate(manager.samples_per_channel);
        out
    })
}

/// Forward FFT both signals (R2C) and return their half-spectrum pointwise
/// product `H1 * conj(H2)`. Output length is `manager.half_spectrum_size()`.
/// Retained as a public helper for tests and any external integrations; the
/// production xcorr path inlines this work.
pub fn calculate_fft_pointwise_product(
    signal_1: &[f64],
    signal_2: &[f64],
    manager: &mut FftManager,
    _fft_points: usize,
) -> Vec<Complex64> {
    let half = manager.half_spectrum_size();
    let mut h1 = vec![Complex64::zero(); half];
    let mut h2 = vec![Complex64::zero(); half];
    manager.forward_r2c(signal_1, &mut h1);
    manager.forward_r2c(signal_2, &mut h2);
    for (a, b) in h1.iter_mut().zip(h2.iter()) {
        let ar = a.re;
        let ai = a.im;
        let br = b.re;
        let bi = b.im;
        a.re = ar * br + ai * bi;
        a.im = ai * br - ar * bi;
    }
    h1
}

///
/// Returns the mantissa and the exponent of a given floating point value.
pub fn frexp(s: f64) -> (f64, i32) {
    if 0.0 == s {
        (s, 0)
    } else {
        let lg = s.abs().log2();
        let x = (lg - lg.floor() - 1.0).exp2();
        let exp = lg.floor() + 1.0;
        (s.signum() * x, exp as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn best_lag_signals_have_equal_length() {
        let ref_signal = vec![
            2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0, 2.0, 2.0,
        ];
        let deg_signal_lag2 = vec![
            1.2, 0.1, -3.3, 0.1, 1.1, 2.2, 2.1, 7.1, 8.3, 6.8, 2.4, 2.2, 2.2, 2.1,
        ];

        assert_eq!(deg_signal_lag2.len(), 14);
        let ref_signal_mat = Array1::from_vec(ref_signal);
        let deg_signal_lag2_mat = Array1::from_vec(deg_signal_lag2);
        assert_eq!(ref_signal_mat.len(), deg_signal_lag2_mat.len());
        let best_lag = calculate_best_lag(
            ref_signal_mat.as_slice().unwrap(),
            deg_signal_lag2_mat.as_slice().unwrap(),
        )
        .unwrap();

        let expected_result = 2;
        assert_eq!(best_lag, expected_result);
    }

    #[test]
    fn best_lag_reference_is_shorter_than_degraded() {
        let ref_signal = vec![
            2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0, 2.0, 2.0,
        ];
        let deg_signal_lag2 = vec![
            1.2, 0.1, -3.3, 0.1, 1.1, 2.2, 2.1, 7.1, 8.3, 6.8, 2.4, 2.2, 2.2, 2.1, 2.0,
        ];

        assert!(ref_signal.len() < deg_signal_lag2.len());
        let ref_signal_mat = Array1::from_vec(ref_signal);
        let deg_signal_lag2_mat = Array1::from_vec(deg_signal_lag2);
        let best_lag = calculate_best_lag(
            ref_signal_mat.as_slice().unwrap(),
            deg_signal_lag2_mat.as_slice().unwrap(),
        )
        .unwrap();

        let expected_result = 2;
        assert_eq!(best_lag, expected_result);
    }

    #[test]
    fn best_lag_reference_is_longer_than_degraded() {
        let ref_signal = vec![
            2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0, 2.0, 2.0,
        ];
        let deg_signal_lag2 = vec![
            1.2, 0.1, -3.3, 0.1, 1.1, 2.2, 2.1, 7.1, 8.3, 6.8, 2.4, 2.2, 2.2,
        ];
        assert!(ref_signal.len() > deg_signal_lag2.len());

        let ref_signal_mat = Array1::from_vec(ref_signal);
        let deg_signal_lag2_mat = Array1::from_vec(deg_signal_lag2);
        let best_lag = calculate_best_lag(
            ref_signal_mat.as_slice().unwrap(),
            deg_signal_lag2_mat.as_slice().unwrap(),
        )
        .unwrap();

        let expected_result = 2;
        assert_eq!(best_lag, expected_result);
    }
    #[test]
    fn best_lag_is_negative() {
        let ref_signal = vec![
            2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0, 2.0, 2.0,
        ];
        let deg_signal_lag2 = vec![
            2.0, 2.0, 2.0, 2.0, 1.0, 0.1, -3.0, 0.1, 1.0, 2.0, 2.0, 6.0, 8.0, 6.0,
        ];

        let ref_signal_mat = Array1::from_vec(ref_signal);
        let deg_signal_lag2_mat = Array1::from_vec(deg_signal_lag2);
        let best_lag = calculate_best_lag(
            ref_signal_mat.as_slice().unwrap(),
            deg_signal_lag2_mat.as_slice().unwrap(),
        )
        .unwrap();

        let expected_result = -2;
        assert_eq!(best_lag, expected_result);
    }

    #[test]
    fn test_frexp() {
        let (_, result) = frexp(27.0f64);
        let expected_result = 5;

        assert_eq!(result, expected_result);
    }
}
