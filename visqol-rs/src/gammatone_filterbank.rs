use crate::{constants, signal_filter};
use std::simd::f64x2;

/// Bank of gammatone filters on each frame of a time domain signal to construct a spectrogram representation.
/// This implementation is fixed to a 4th order filterbank.
pub struct GammatoneFilterbank<const NUM_BANDS: usize> {
    pub min_freq: f64,

    filter_conditions_1: [[f64; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],
    filter_conditions_2: [[f64; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],
    filter_conditions_3: [[f64; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],
    filter_conditions_4: [[f64; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],

    filter_coeff_a0: Vec<f64>,
    filter_coeff_a11: Vec<f64>,
    filter_coeff_a12: Vec<f64>,
    filter_coeff_a13: Vec<f64>,
    filter_coeff_a14: Vec<f64>,
    filter_coeff_a2: Vec<f64>,
    filter_coeff_b0: Vec<f64>,
    filter_coeff_b1: Vec<f64>,
    filter_coeff_b2: Vec<f64>,
    filter_coeff_gain: Vec<f64>,

    // Pre-allocated ping-pong buffers for intermediate filter results.
    // Avoids allocating 4 Vecs per band per frame.
    buf_a: Vec<f64>,
    buf_b: Vec<f64>,
}

impl<const NUM_BANDS: usize> GammatoneFilterbank<NUM_BANDS> {
    /// Creates a new gammatone filterbank with the desired number of frequency bands and the minimum frequency.
    pub fn new(min_freq: f64) -> Self {
        Self {
            min_freq,
            filter_conditions_1: [[0.0; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],
            filter_conditions_2: [[0.0; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],
            filter_conditions_3: [[0.0; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],
            filter_conditions_4: [[0.0; constants::NUM_FILTER_CONDITIONS]; NUM_BANDS],
            filter_coeff_a0: Vec::new(),
            filter_coeff_a11: Vec::new(),
            filter_coeff_a12: Vec::new(),
            filter_coeff_a13: Vec::new(),
            filter_coeff_a14: Vec::new(),
            filter_coeff_a2: Vec::new(),
            filter_coeff_b0: Vec::new(),
            filter_coeff_b1: Vec::new(),
            filter_coeff_b2: Vec::new(),
            filter_coeff_gain: Vec::new(),
            buf_a: Vec::new(),
            buf_b: Vec::new(),
        }
    }

    /// Sets all internal states of the filterbank to 0.
    pub fn reset_filter_conditions(&mut self) {
        self.filter_conditions_1 = [[0.0, 0.0]; NUM_BANDS];
        self.filter_conditions_2 = [[0.0, 0.0]; NUM_BANDS];
        self.filter_conditions_3 = [[0.0, 0.0]; NUM_BANDS];
        self.filter_conditions_4 = [[0.0, 0.0]; NUM_BANDS];
    }

    /// Populates the filter coefficients with `filter_coeffs`.
    pub fn set_filter_coefficients(&mut self, filter_coeffs: &ndarray::Array2<f64>) {
        self.filter_coeff_a0 = filter_coeffs.column(0).to_vec();
        self.filter_coeff_a11 = filter_coeffs.column(1).to_vec();
        self.filter_coeff_a12 = filter_coeffs.column(2).to_vec();
        self.filter_coeff_a13 = filter_coeffs.column(3).to_vec();
        self.filter_coeff_a14 = filter_coeffs.column(4).to_vec();
        self.filter_coeff_a2 = filter_coeffs.column(5).to_vec();
        self.filter_coeff_b0 = filter_coeffs.column(6).to_vec();
        self.filter_coeff_b1 = filter_coeffs.column(7).to_vec();
        self.filter_coeff_b2 = filter_coeffs.column(8).to_vec();
        self.filter_coeff_gain = filter_coeffs.column(9).to_vec();
    }

    /// Ensures internal ping-pong buffers are large enough for `len` samples.
    fn ensure_buffers(&mut self, len: usize) {
        if self.buf_a.len() < len {
            self.buf_a.resize(len, 0.0);
            self.buf_b.resize(len, 0.0);
        }
    }

    /// Applies the gammatone filterbank on the time-domain signal `signal`,
    /// producing a Gammatone spectrogram.
    ///
    /// Uses pre-allocated ping-pong buffers to avoid per-band heap allocation.
    #[inline(always)]
    pub fn apply_filter(&mut self, input_signal: &[f64]) -> ndarray::Array2<f64> {
        let sig_len = input_signal.len();
        self.ensure_buffers(sig_len);

        let mut output = ndarray::Array2::<f64>::zeros((NUM_BANDS, sig_len));

        for band in 0..NUM_BANDS {
            let gain_inv = 1.0 / self.filter_coeff_gain[band];
            let a1 = [
                self.filter_coeff_a0[band] * gain_inv,
                self.filter_coeff_a11[band] * gain_inv,
                self.filter_coeff_a2[band] * gain_inv,
            ];
            let a2 = [
                self.filter_coeff_a0[band],
                self.filter_coeff_a12[band],
                self.filter_coeff_a2[band],
            ];
            let a3 = [
                self.filter_coeff_a0[band],
                self.filter_coeff_a13[band],
                self.filter_coeff_a2[band],
            ];
            let a4 = [
                self.filter_coeff_a0[band],
                self.filter_coeff_a14[band],
                self.filter_coeff_a2[band],
            ];
            let b = [
                self.filter_coeff_b0[band],
                self.filter_coeff_b1[band],
                self.filter_coeff_b2[band],
            ];

            // 1st filter: input -> buf_a
            signal_filter::filter_signal_into(
                &a1,
                &b,
                input_signal,
                &mut self.buf_a[..sig_len],
                &mut self.filter_conditions_1[band],
            );

            // 2nd filter: buf_a -> buf_b
            signal_filter::filter_signal_into(
                &a2,
                &b,
                &self.buf_a[..sig_len],
                &mut self.buf_b[..sig_len],
                &mut self.filter_conditions_2[band],
            );

            // 3rd filter: buf_b -> buf_a
            signal_filter::filter_signal_into(
                &a3,
                &b,
                &self.buf_b[..sig_len],
                &mut self.buf_a[..sig_len],
                &mut self.filter_conditions_3[band],
            );

            // 4th filter: buf_a -> output row directly
            {
                let out_row = output.row_mut(band);
                let out_slice = out_row.into_slice().expect("output row not contiguous");
                signal_filter::filter_signal_into(
                    &a4,
                    &b,
                    &self.buf_a[..sig_len],
                    out_slice,
                    &mut self.filter_conditions_4[band],
                );
            }
        }
        output
    }

    /// Applies the filterbank and computes per-band RMS in a single pass.
    ///
    /// Processes bands in pairs using `f64x2` portable SIMD. For each pair,
    /// all 4 cascaded IIR filter stages are run sample-by-sample with both
    /// bands packed into SIMD lanes, and sum-of-squares is accumulated
    /// inline in the final stage. This gives ~2× throughput on the IIR
    /// cascade (the dominant cost).
    #[inline(always)]
    pub fn apply_filter_rms(&mut self, input_signal: &[f64], rms_out: &mut [f64]) {
        let sig_len = input_signal.len();
        let inv_len = 1.0 / sig_len as f64;

        // Process pairs of bands with SIMD
        let num_pairs = NUM_BANDS / 2;
        for pair in 0..num_pairs {
            let b0 = pair * 2;
            let b1 = b0 + 1;

            self.apply_filter_pair_rms(b0, b1, input_signal, inv_len, rms_out);
        }

        // Handle odd trailing band with scalar path
        if NUM_BANDS % 2 == 1 {
            let band = NUM_BANDS - 1;
            self.ensure_buffers(sig_len);
            self.apply_filter_single_rms(band, input_signal, sig_len, inv_len, rms_out);
        }
    }

    /// Process a pair of bands through all 4 cascaded IIR stages using f64x2.
    /// Each sample goes through stage1→stage2→stage3→stage4 for both bands
    /// simultaneously in SIMD registers. No intermediate buffers needed.
    #[inline(always)]
    fn apply_filter_pair_rms(
        &mut self,
        b0: usize,
        b1: usize,
        input_signal: &[f64],
        inv_len: f64,
        rms_out: &mut [f64],
    ) {
        let gi0 = 1.0 / self.filter_coeff_gain[b0];
        let gi1 = 1.0 / self.filter_coeff_gain[b1];

        // Stage 1 (numerator has gain normalization)
        let s1_n0 = f64x2::from_array([self.filter_coeff_a0[b0] * gi0, self.filter_coeff_a0[b1] * gi1]);
        let s1_n1 = f64x2::from_array([self.filter_coeff_a11[b0] * gi0, self.filter_coeff_a11[b1] * gi1]);
        let s1_n2 = f64x2::from_array([self.filter_coeff_a2[b0] * gi0, self.filter_coeff_a2[b1] * gi1]);
        // Stages 2-4 share a0 and a2
        let a0 = f64x2::from_array([self.filter_coeff_a0[b0], self.filter_coeff_a0[b1]]);
        let a2 = f64x2::from_array([self.filter_coeff_a2[b0], self.filter_coeff_a2[b1]]);
        let s2_n1 = f64x2::from_array([self.filter_coeff_a12[b0], self.filter_coeff_a12[b1]]);
        let s3_n1 = f64x2::from_array([self.filter_coeff_a13[b0], self.filter_coeff_a13[b1]]);
        let s4_n1 = f64x2::from_array([self.filter_coeff_a14[b0], self.filter_coeff_a14[b1]]);
        // Denominator (same for all 4 stages)
        let d1 = f64x2::from_array([self.filter_coeff_b1[b0], self.filter_coeff_b1[b1]]);
        let d2 = f64x2::from_array([self.filter_coeff_b2[b0], self.filter_coeff_b2[b1]]);

        // Filter state: 4 stages × 2 conditions
        let mut s1c0 = f64x2::from_array([self.filter_conditions_1[b0][0], self.filter_conditions_1[b1][0]]);
        let mut s1c1 = f64x2::from_array([self.filter_conditions_1[b0][1], self.filter_conditions_1[b1][1]]);
        let mut s2c0 = f64x2::from_array([self.filter_conditions_2[b0][0], self.filter_conditions_2[b1][0]]);
        let mut s2c1 = f64x2::from_array([self.filter_conditions_2[b0][1], self.filter_conditions_2[b1][1]]);
        let mut s3c0 = f64x2::from_array([self.filter_conditions_3[b0][0], self.filter_conditions_3[b1][0]]);
        let mut s3c1 = f64x2::from_array([self.filter_conditions_3[b0][1], self.filter_conditions_3[b1][1]]);
        let mut s4c0 = f64x2::from_array([self.filter_conditions_4[b0][0], self.filter_conditions_4[b1][0]]);
        let mut s4c1 = f64x2::from_array([self.filter_conditions_4[b0][1], self.filter_conditions_4[b1][1]]);

        let mut sum_sq = f64x2::splat(0.0);

        for &s in input_signal {
            let sv = f64x2::splat(s);

            // Stage 1
            let f1 = s1_n0 * sv + s1c0;
            s1c0 = s1_n1 * sv + s1c1 - d1 * f1;
            s1c1 = s1_n2 * sv - d2 * f1;

            // Stage 2 (input = f1)
            let f2 = a0 * f1 + s2c0;
            s2c0 = s2_n1 * f1 + s2c1 - d1 * f2;
            s2c1 = a2 * f1 - d2 * f2;

            // Stage 3 (input = f2)
            let f3 = a0 * f2 + s3c0;
            s3c0 = s3_n1 * f2 + s3c1 - d1 * f3;
            s3c1 = a2 * f2 - d2 * f3;

            // Stage 4 (input = f3) + accumulate sum-of-squares
            let f4 = a0 * f3 + s4c0;
            s4c0 = s4_n1 * f3 + s4c1 - d1 * f4;
            s4c1 = a2 * f3 - d2 * f4;

            sum_sq += f4 * f4;
        }

        // Write back filter state
        let s1c0a = s1c0.to_array();
        let s1c1a = s1c1.to_array();
        self.filter_conditions_1[b0] = [s1c0a[0], s1c1a[0]];
        self.filter_conditions_1[b1] = [s1c0a[1], s1c1a[1]];
        let s2c0a = s2c0.to_array();
        let s2c1a = s2c1.to_array();
        self.filter_conditions_2[b0] = [s2c0a[0], s2c1a[0]];
        self.filter_conditions_2[b1] = [s2c0a[1], s2c1a[1]];
        let s3c0a = s3c0.to_array();
        let s3c1a = s3c1.to_array();
        self.filter_conditions_3[b0] = [s3c0a[0], s3c1a[0]];
        self.filter_conditions_3[b1] = [s3c0a[1], s3c1a[1]];
        let s4c0a = s4c0.to_array();
        let s4c1a = s4c1.to_array();
        self.filter_conditions_4[b0] = [s4c0a[0], s4c1a[0]];
        self.filter_conditions_4[b1] = [s4c0a[1], s4c1a[1]];

        // Extract per-band RMS
        let sq = sum_sq.to_array();
        rms_out[b0] = (sq[0] * inv_len).sqrt();
        rms_out[b1] = (sq[1] * inv_len).sqrt();
    }

    /// Scalar fallback for a single band.
    #[inline(always)]
    fn apply_filter_single_rms(
        &mut self,
        band: usize,
        input_signal: &[f64],
        sig_len: usize,
        inv_len: f64,
        rms_out: &mut [f64],
    ) {
        let gain_inv = 1.0 / self.filter_coeff_gain[band];
        let a1 = [
            self.filter_coeff_a0[band] * gain_inv,
            self.filter_coeff_a11[band] * gain_inv,
            self.filter_coeff_a2[band] * gain_inv,
        ];
        let a2 = [self.filter_coeff_a0[band], self.filter_coeff_a12[band], self.filter_coeff_a2[band]];
        let a3 = [self.filter_coeff_a0[band], self.filter_coeff_a13[band], self.filter_coeff_a2[band]];
        let a4 = [self.filter_coeff_a0[band], self.filter_coeff_a14[band], self.filter_coeff_a2[band]];
        let b = [self.filter_coeff_b0[band], self.filter_coeff_b1[band], self.filter_coeff_b2[band]];

        signal_filter::filter_signal_into(&a1, &b, input_signal, &mut self.buf_a[..sig_len], &mut self.filter_conditions_1[band]);
        signal_filter::filter_signal_into(&a2, &b, &self.buf_a[..sig_len], &mut self.buf_b[..sig_len], &mut self.filter_conditions_2[band]);
        signal_filter::filter_signal_into(&a3, &b, &self.buf_b[..sig_len], &mut self.buf_a[..sig_len], &mut self.filter_conditions_3[band]);
        signal_filter::filter_signal_into(&a4, &b, &self.buf_a[..sig_len], &mut self.buf_b[..sig_len], &mut self.filter_conditions_4[band]);

        let mut sq = 0.0f64;
        for &x in &self.buf_b[..sig_len] { sq += x * x; }
        rms_out[band] = (sq * inv_len).sqrt();
    }
}

#[cfg(test)]
mod tests {
    use crate::equivalent_rectangular_bandwidth;
    use approx::assert_abs_diff_eq;
    use ndarray::Axis;

    use super::*;
    #[test]
    fn gammatone_filterbank() {
        let fs = 48000;
        const NUM_BANDS: usize = 32;
        let min_freq = 50.0f64;

        let ten_samples = vec![0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.5, 0.7, 0.9];

        let (mut filter_coeffs, _) = equivalent_rectangular_bandwidth::make_filters::<NUM_BANDS>(
            fs,
            min_freq,
            fs as f64 / 2.0,
        );

        filter_coeffs.invert_axis(Axis(0));

        let epsilon = 0.0001;

        // Check if filtering works as intended.
        let mut filterbank = GammatoneFilterbank::<{ NUM_BANDS }>::new(min_freq);
        filterbank.reset_filter_conditions();
        filterbank.set_filter_coefficients(&filter_coeffs);

        let filtered_signal = filterbank.apply_filter(&ten_samples);

        // Check dimensions
        assert_eq!(filtered_signal.ncols(), 10);
        assert_eq!(filtered_signal.nrows(), 32);

        // Check individual elements
        let expected_output = [1.028e-10, 6.15143e-10, 2.14718e-09];

        for (&res, ex) in expected_output.iter().zip(filtered_signal) {
            assert_abs_diff_eq!(res, ex, epsilon = epsilon);
        }
    }
}
