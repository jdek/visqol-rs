use crate::fft_manager::FftManager;
use num::complex::Complex64;
use num::Zero;

/// Calculate the maximum delay between two signals.
pub fn calculate_best_lag(signal_1: &[f64], signal_2: &[f64]) -> Option<i64> {
    let biggest = signal_1.len().max(signal_2.len());
    let max_lag = (biggest - 1) as i64;
    let (_, exp) = frexp((biggest * 2 - 1) as f64);
    let fft_points = 2usize.pow(exp as u32);

    // Owned copies so the inputs can be zero-padded inside the manager call.
    let mut sig1 = signal_1.to_vec();
    let mut sig2 = signal_2.to_vec();
    sig1.resize(biggest, 0.0);
    sig2.resize(biggest, 0.0);

    Some(FftManager::with_cached(fft_points, |manager| {
        manager.xcorr_inverse_pointwise(&sig1, &sig2, |corr| {
            // Correlation output layout:
            //   [0..max_lag+1]  = positive lags (index k → lag k)
            //   [n-max_lag..n]  = negative lags (index n-k → lag -k)
            let n = corr.len();
            let neg_start = n - max_lag as usize;
            let mut best_abs = f64::MIN;
            let mut best_lag_result: i64 = 0;
            for (i, &v) in corr[neg_start..n].iter().enumerate() {
                let abs_val = v.abs();
                if abs_val > best_abs {
                    best_abs = abs_val;
                    best_lag_result = (neg_start + i) as i64 - n as i64;
                }
            }
            for (i, &v) in corr[..=max_lag as usize].iter().enumerate() {
                let abs_val = v.abs();
                if abs_val > best_abs {
                    best_abs = abs_val;
                    best_lag_result = i as i64;
                }
            }
            best_lag_result
        })
    }))
}

/// Calculates the pointwise inverse fft product of 2 signals
pub fn calculate_inverse_fft_pointwise_product(
    signal_1: &mut Vec<f64>,
    signal_2: &mut Vec<f64>,
) -> Vec<f64> {
    let biggest = signal_1.len().max(signal_2.len());
    signal_1.resize(biggest, 0.0);
    signal_2.resize(biggest, 0.0);
    let (_, exp) = frexp((biggest * 2 - 1) as f64);
    let fft_points = 2usize.pow(exp as u32);
    FftManager::with_cached(fft_points, |manager| {
        manager.xcorr_inverse_pointwise(signal_1, signal_2, |corr| corr.to_vec())
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
