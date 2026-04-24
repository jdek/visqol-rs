use crate::audio_signal::AudioSignal;
use crate::envelope;
use crate::perf_trace;
use crate::xcorr;
use ndarray::Array1;
use ndarray::{concatenate, s, Axis};

/// Aligns and truncates so ref and deg share a length. Matches upstream
/// C++/Python `Alignment::AlignAndTruncate`.
pub fn align_and_truncate(
    ref_signal: &AudioSignal,
    deg_signal: &AudioSignal,
) -> Option<(AudioSignal, AudioSignal, f64)> {
    let (aligned_deg, lag_seconds) = globally_align(ref_signal, deg_signal)?;

    let ref_len = ref_signal.data_matrix.len();
    let deg_len = aligned_deg.data_matrix.len();

    let (ref_data, deg_data) = if ref_len > deg_len {
        let r = ref_signal
            .data_matrix
            .slice(s![..deg_len])
            .to_owned();
        (r, aligned_deg.data_matrix)
    } else if ref_len < deg_len {
        let lag_samples = (lag_seconds * ref_signal.sample_rate as f64) as i64;
        if lag_samples > 0 {
            let ls = lag_samples as usize;
            let r = ref_signal.data_matrix.slice(s![ls..]).to_owned();
            let d = aligned_deg.data_matrix
                .slice(s![ls..ls + r.len()])
                .to_owned();
            (r, d)
        } else {
            let d = aligned_deg.data_matrix.slice(s![..ref_len]).to_owned();
            (ref_signal.data_matrix.clone(), d)
        }
    } else {
        (ref_signal.data_matrix.clone(), aligned_deg.data_matrix)
    };

    let min_len = ref_data.len().min(deg_data.len());
    let ref_data = ref_data.slice(s![..min_len]).to_owned();
    let deg_data = deg_data.slice(s![..min_len]).to_owned();

    Some((
        AudioSignal::from_owned(ref_data, ref_signal.sample_rate),
        AudioSignal::from_owned(deg_data, deg_signal.sample_rate),
        lag_seconds,
    ))
}

/// Globally aligns the degraded signal to the reference using upper-envelope
/// cross-correlation. Matches upstream C++/Python `Alignment::GloballyAlign`.
pub fn globally_align(
    ref_signal: &AudioSignal,
    deg_signal: &AudioSignal,
) -> Option<(AudioSignal, f64)> {
    let _t = perf_trace::span("globally_align");
    let ref_env = {
        let _t = perf_trace::span("globally_align.envelope");
        envelope::calculate_upper_env(&ref_signal.data_matrix)?
    };
    let deg_env = {
        let _t = perf_trace::span("globally_align.envelope");
        envelope::calculate_upper_env(&deg_signal.data_matrix)?
    };

    let best_lag = {
        let _t = perf_trace::span("globally_align.xcorr");
        xcorr::calculate_best_lag(ref_env.as_slice()?, deg_env.as_slice()?)?
    };

    if best_lag == 0 || best_lag.abs() > (ref_signal.data_matrix.len() / 2) as i64 {
        let new_deg_signal =
            AudioSignal::from_owned(deg_signal.data_matrix.clone(), deg_signal.sample_rate);
        return Some((new_deg_signal, 0.0f64));
    }

    let new_deg_matrix = if best_lag < 0 {
        deg_signal
            .data_matrix
            .slice(s![best_lag.unsigned_abs() as usize..deg_signal.data_matrix.len()])
            .to_owned()
    } else {
        let zeros = Array1::<f64>::zeros(best_lag as usize);
        concatenate(Axis(0), &[zeros.view(), deg_signal.data_matrix.view()])
            .expect("Failed to zero pad degraded matrix!")
    };

    Some((
        AudioSignal::from_owned(new_deg_matrix, deg_signal.sample_rate),
        best_lag as f64 / deg_signal.sample_rate as f64,
    ))
}
