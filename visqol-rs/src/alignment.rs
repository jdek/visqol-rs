use crate::audio_signal::AudioSignal;
use crate::xcorr;
use ndarray::Array1;
use ndarray::{concatenate, s, Axis};

/// Creates copy of `deg_signal` which is time-aligned to `ref_signal` using
/// direct cross-correlation (without Hilbert envelope extraction).
/// Returns a copy of the reference signal, a copy of the aligned degraded signal and the delay between the signals.
pub fn align_and_truncate(
    ref_signal: &AudioSignal,
    deg_signal: &AudioSignal,
) -> Option<(AudioSignal, AudioSignal, f64)> {
    let best_lag = xcorr::calculate_best_lag(
        ref_signal.data_matrix.as_slice()?,
        deg_signal.data_matrix.as_slice()?,
    )?;

    if best_lag == 0 || best_lag.abs() > (ref_signal.data_matrix.len() / 2) as i64 {
        return Some((
            AudioSignal::from_owned(ref_signal.data_matrix.clone(), ref_signal.sample_rate),
            AudioSignal::from_owned(deg_signal.data_matrix.clone(), deg_signal.sample_rate),
            0.0,
        ));
    }

    let lag_f = best_lag as f64 / deg_signal.sample_rate as f64;

    // Align degraded signal
    let new_deg_matrix = if best_lag < 0 {
        deg_signal.data_matrix
            .slice(s![best_lag.unsigned_abs() as usize..deg_signal.data_matrix.len()])
            .to_owned()
    } else {
        let zeros = Array1::<f64>::zeros(best_lag as usize);
        concatenate(Axis(0), &[zeros.view(), deg_signal.data_matrix.view()])
            .expect("Failed to zero pad degraded matrix!")
    };

    // Truncate to same length
    let (new_ref_matrix, new_deg_final) = match ref_signal.data_matrix.len().cmp(&new_deg_matrix.len()) {
        std::cmp::Ordering::Less => {
            let lag_samples = (lag_f * ref_signal.sample_rate as f64) as usize;
            let r = ref_signal.data_matrix
                .slice(s![lag_samples..ref_signal.len()])
                .to_owned();
            let d = new_deg_matrix
                .slice(s![lag_samples..ref_signal.len()])
                .to_owned();
            (r, d)
        }
        std::cmp::Ordering::Greater => {
            let deg_len = new_deg_matrix.len();
            let r = ref_signal.data_matrix.slice(s![..deg_len]).to_owned();
            (r, new_deg_matrix)
        }
        std::cmp::Ordering::Equal => {
            (ref_signal.data_matrix.clone(), new_deg_matrix)
        }
    };

    Some((
        AudioSignal::from_owned(new_ref_matrix, ref_signal.sample_rate),
        AudioSignal::from_owned(new_deg_final, deg_signal.sample_rate),
        lag_f,
    ))
}

/// Aligns a degraded signal to the reference signal using direct
/// cross-correlation. Returns the aligned degraded signal and the lag in seconds.
pub fn globally_align(
    ref_signal: &AudioSignal,
    deg_signal: &AudioSignal,
) -> Option<(AudioSignal, f64)> {
    let best_lag = xcorr::calculate_best_lag(
        ref_signal.data_matrix.as_slice()?,
        deg_signal.data_matrix.as_slice()?,
    )?;

    if best_lag == 0 || best_lag.abs() > (ref_signal.data_matrix.len() / 2) as i64 {
        let new_deg_signal =
            AudioSignal::from_owned(deg_signal.data_matrix.clone(), deg_signal.sample_rate);
        Some((new_deg_signal, 0.0f64))
    } else {
        let new_deg_matrix = if best_lag < 0 {
            deg_signal.data_matrix
                .slice(s![best_lag.unsigned_abs() as usize..deg_signal.data_matrix.len()])
                .to_owned()
        } else {
            let zeros = Array1::<f64>::zeros(best_lag as usize);
            concatenate(Axis(0), &[zeros.view(), deg_signal.data_matrix.view()])
                .expect("Failed to zero pad degraded matrix!")
        };

        let new_deg_signal = AudioSignal::from_owned(new_deg_matrix, deg_signal.sample_rate);
        Some((
            new_deg_signal,
            (best_lag as f64 / deg_signal.sample_rate as f64),
        ))
    }
}
