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
            AudioSignal::new(ref_signal.data_matrix.as_slice()?, ref_signal.sample_rate),
            AudioSignal::new(deg_signal.data_matrix.as_slice()?, deg_signal.sample_rate),
            0.0,
        ));
    }

    let lag_f = best_lag as f64 / deg_signal.sample_rate as f64;

    // Align degraded signal
    let mut new_deg_matrix = deg_signal.data_matrix.clone();
    if best_lag < 0 {
        new_deg_matrix = new_deg_matrix
            .slice(s![best_lag.unsigned_abs() as usize..deg_signal.data_matrix.len()])
            .to_owned();
    } else {
        let zeros = Array1::<f64>::zeros(best_lag as usize);
        new_deg_matrix = concatenate(Axis(0), &[zeros.view(), new_deg_matrix.view()])
            .expect("Failed to zero pad degraded matrix!");
    }

    // Truncate to same length
    let mut new_ref_matrix = ref_signal.data_matrix.clone();
    match new_ref_matrix.len().cmp(&new_deg_matrix.len()) {
        std::cmp::Ordering::Less => {
            let lag_samples = (lag_f * ref_signal.sample_rate as f64) as usize;
            new_ref_matrix = new_ref_matrix
                .slice(s![lag_samples..ref_signal.len()])
                .to_owned();
            new_deg_matrix = new_deg_matrix
                .slice(s![lag_samples..ref_signal.len()])
                .to_owned();
        }
        std::cmp::Ordering::Greater => {
            new_ref_matrix = new_ref_matrix.slice(s![..new_deg_matrix.len()]).to_owned();
        }
        _ => (),
    }

    Some((
        AudioSignal::new(new_ref_matrix.as_slice()?, ref_signal.sample_rate),
        AudioSignal::new(new_deg_matrix.as_slice()?, deg_signal.sample_rate),
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
            AudioSignal::new(deg_signal.data_matrix.as_slice()?, deg_signal.sample_rate);
        Some((new_deg_signal, 0.0f64))
    } else {
        let mut new_deg_matrix = deg_signal.data_matrix.clone();
        if best_lag < 0 {
            new_deg_matrix = new_deg_matrix
                .slice(s![
                    best_lag.unsigned_abs() as usize..deg_signal.data_matrix.len()
                ])
                .to_owned();
        } else {
            let zeros = Array1::<f64>::zeros(best_lag as usize);
            new_deg_matrix = concatenate(Axis(0), &[zeros.view(), new_deg_matrix.view()])
                .expect("Failed to zero pad degraded matrix!");
        }

        let new_deg_signal = AudioSignal::new(
            new_deg_matrix
                .as_slice()
                .expect("Failed to create AudioSignal from slice!"),
            deg_signal.sample_rate,
        );
        Some((
            new_deg_signal,
            (best_lag as f64 / deg_signal.sample_rate as f64),
        ))
    }
}
