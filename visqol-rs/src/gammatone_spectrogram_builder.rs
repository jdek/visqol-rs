use crate::analysis_window::AnalysisWindow;
use crate::constants::NUM_BANDS_SPEECH;
use crate::equivalent_rectangular_bandwidth;
use crate::gammatone_filterbank::GammatoneFilterbank;
use crate::spectrogram::Spectrogram;
use crate::spectrogram_builder::SpectrogramBuilder;
use crate::{audio_signal::AudioSignal, visqol_error::VisqolError};
use ndarray::{Array2, Axis};

/// Produces a frequency domain representation from a time domain signal using a gammatone filterbank.
pub struct GammatoneSpectrogramBuilder<const NUM_BANDS: usize> {
    filter_bank: GammatoneFilterbank<NUM_BANDS>,
    /// Cached filter coefficients + sorted center freqs, keyed by (sample_rate, max_freq).
    cached_coeffs: Option<CachedCoeffs>,
}

struct CachedCoeffs {
    sample_rate: usize,
    max_freq: f64,
    filter_coeffs: ndarray::Array2<f64>,
    center_freqs: Vec<f64>,
}

impl<const NUM_BANDS: usize> SpectrogramBuilder for GammatoneSpectrogramBuilder<NUM_BANDS> {
    fn build(
        &mut self,
        signal: &AudioSignal,
        window: &AnalysisWindow,
    ) -> Result<Spectrogram, VisqolError> {
        let time_domain_signal = &signal.data_matrix;
        let sample_rate = signal.sample_rate;
        let max_freq = if NUM_BANDS == NUM_BANDS_SPEECH {
            Self::SPEECH_MODE_MAX_FREQ as f64
        } else {
            sample_rate as f64 / 2.0
        };

        // Cache or reuse filter coefficients
        let need_recompute = match &self.cached_coeffs {
            Some(c) => c.sample_rate != sample_rate as usize || c.max_freq != max_freq,
            None => true,
        };

        if need_recompute {
            let (mut filter_coeffs, mut center_freqs) =
                equivalent_rectangular_bandwidth::make_filters::<NUM_BANDS>(
                    sample_rate as usize,
                    self.filter_bank.min_freq,
                    max_freq,
                );
            filter_coeffs.invert_axis(Axis(0));
            center_freqs.sort_by(|a, b| a.partial_cmp(b).expect("Failed to sort center frequencies!"));
            self.cached_coeffs = Some(CachedCoeffs {
                sample_rate: sample_rate as usize,
                max_freq,
                filter_coeffs,
                center_freqs,
            });
        }

        let cached = self.cached_coeffs.as_ref().unwrap();
        if need_recompute {
            self.filter_bank.set_filter_coefficients(&cached.filter_coeffs);
        }

        let hop_size = (window.size as f64 * window.overlap) as usize;

        if time_domain_signal.len() < window.size {
            return Err(VisqolError::TooFewSamples {
                found: time_domain_signal.len(),
                minimum_required: window.size,
            });
        }

        let num_cols = 1 + ((time_domain_signal.len() - window.size) / hop_size);
        let mut out_matrix = Array2::<f64>::zeros((NUM_BANDS, num_cols));

        // Pre-allocate a column buffer for RMS results
        let mut rms_col = vec![0.0f64; NUM_BANDS];
        // Scratch buffer for the Hann-windowed frame
        let mut windowed_frame = vec![0.0f64; window.size];

        let signal_slice = time_domain_signal
            .as_slice()
            .expect("Failed to convert audio signal to slice");

        for (index, frame_start) in (0..=signal_slice.len() - window.size)
            .step_by(hop_size)
            .enumerate()
        {
            let frame = &signal_slice[frame_start..frame_start + window.size];
            // Apply Hann window (matches upstream visqol C++/Python)
            for ((w, &s), &h) in windowed_frame
                .iter_mut()
                .zip(frame.iter())
                .zip(window.hann.iter())
            {
                *w = s * h;
            }

            self.filter_bank
                .apply_filter_rms_fresh(&windowed_frame, &mut rms_col);

            // Write RMS values directly into the output column
            for j in 0..NUM_BANDS {
                out_matrix[(j, index)] = rms_col[j];
            }
        }

        Ok(Spectrogram::new(out_matrix, cached.center_freqs.clone()))
    }
}

impl<const NUM_BANDS: usize> GammatoneSpectrogramBuilder<NUM_BANDS> {
    const SPEECH_MODE_MAX_FREQ: u32 = 8000;

    /// Creates a new gammatone spectrogram builder with the given gammatone filterbank.
    pub fn new(filter_bank: GammatoneFilterbank<NUM_BANDS>) -> Self {
        Self {
            filter_bank,
            cached_coeffs: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis_window::AnalysisWindow;
    use crate::audio_utils;
    use crate::gammatone_filterbank::GammatoneFilterbank;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_spec_builder() {
        // Fixed parameters
        const MINIMUM_FREQ: f64 = 50.0;
        const NUM_BANDS: usize = 32;
        const OVERLAP: f64 = 0.25;

        const REF_SPECTRO_NUM_COLS: usize = 802;

        let signal_ref = audio_utils::load_as_mono(
            "test_data/conformance_testdata_subset/contrabassoon48_stereo.wav",
        )
        .unwrap();
        let filter_bank = GammatoneFilterbank::<{ NUM_BANDS }>::new(MINIMUM_FREQ);
        let window = AnalysisWindow::new(signal_ref.sample_rate, OVERLAP, 0.08);

        let mut spectro_builder: GammatoneSpectrogramBuilder<NUM_BANDS> =
            GammatoneSpectrogramBuilder::new(filter_bank);
        let spectrogram_ref = spectro_builder.build(&signal_ref, &window).unwrap();

        // Check 1st element. Hann[0] = 0, so frame 0 is heavily attenuated.
        assert_abs_diff_eq!(spectrogram_ref.data[(0, 0)], 1.363673e-5, epsilon = 1e-9);
        // Check dimensions
        assert_eq!(spectrogram_ref.data.ncols(), REF_SPECTRO_NUM_COLS);
    }
}
