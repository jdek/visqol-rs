use crate::convolution_2d::{conv2d_3x3_reflected_into, perform_valid_2d_conv_with_boundary};
use crate::patch_similarity_comparator::{PatchSimilarityComparator, PatchSimilarityResult};
use ndarray::{arr2, Array1, Array2, Axis, Zip};
use std::sync::LazyLock;

/// Provides a neurogram similarity index measure (NSIM) implementation for a
/// patch similarity comparator. NSIM is a distance metric, adapted from the
/// image processing technique called structural similarity (SSIM) and is here
/// used to compare two patches taken from the reference and degraded
/// spectrograms.
pub struct NeurogramSimiliarityIndexMeasure {
    intensity_range: f64,
}

#[allow(unused)]
impl NeurogramSimiliarityIndexMeasure {
    pub fn new(intensity_range: f64) -> Self { Self { intensity_range } }
}

impl Default for NeurogramSimiliarityIndexMeasure {
    fn default() -> Self {
        Self {
            intensity_range: 1.0,
        }
    }
}

// Pre-computed Gaussian window (constant, same every call)
const W: [[f64; 3]; 3] = [
    [0.0113033910173052, 0.0838251475442633, 0.0113033910173052],
    [0.0838251475442633, 0.619485845753726, 0.0838251475442633],
    [0.0113033910173052, 0.0838251475442633, 0.0113033910173052],
];

static WINDOW: LazyLock<Array2<f64>> = LazyLock::new(|| arr2(&W));

/// Pre-allocated scratch buffers for NSIM computation to avoid per-call allocations.
pub struct NsimScratch {
    mu_ref: Array2<f64>,
    mu_deg: Array2<f64>,
    ref_neuro_sq: Array2<f64>,
    deg_neuro_sq: Array2<f64>,
    ref_neuro_deg: Array2<f64>,
    conv_ref_sq: Array2<f64>,
    conv_deg_sq: Array2<f64>,
    conv_rd: Array2<f64>,
    /// Precomputed sigma_ref_squared = conv(ref²) - mu_ref² (constant per ref patch).
    precomp_srs: Array2<f64>,
    /// Whether precomputed ref values are valid.
    precomp_valid: bool,
}

impl NsimScratch {
    /// Create scratch buffers sized for patches of the given dimensions.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            mu_ref: Array2::zeros((nrows, ncols)),
            mu_deg: Array2::zeros((nrows, ncols)),
            ref_neuro_sq: Array2::zeros((nrows, ncols)),
            deg_neuro_sq: Array2::zeros((nrows, ncols)),
            ref_neuro_deg: Array2::zeros((nrows, ncols)),
            conv_ref_sq: Array2::zeros((nrows, ncols)),
            conv_deg_sq: Array2::zeros((nrows, ncols)),
            conv_rd: Array2::zeros((nrows, ncols)),
            precomp_srs: Array2::zeros((nrows, ncols)),
            precomp_valid: false,
        }
    }

    /// Resize all scratch buffers if the current dimensions don't match.
    pub fn ensure_size(&mut self, nrows: usize, ncols: usize) {
        if self.mu_ref.nrows() != nrows || self.mu_ref.ncols() != ncols {
            *self = Self::new(nrows, ncols);
        }
    }

    /// Precompute reference-only values: mu_ref, conv_ref_sq, and sigma_ref_sq.
    /// Call once per ref_patch before the DP search loop.
    pub fn precompute_ref(&mut self, ref_patch: &Array2<f64>) {
        let window = &*WINDOW;
        conv2d_3x3_reflected_into(window, ref_patch, &mut self.mu_ref);
        Zip::from(&mut self.ref_neuro_sq)
            .and(ref_patch)
            .for_each(|o, &r| *o = r * r);
        conv2d_3x3_reflected_into(window, &self.ref_neuro_sq, &mut self.conv_ref_sq);
        // precomp_srs = conv_ref_sq - mu_ref²
        Zip::from(&mut self.precomp_srs)
            .and(&self.conv_ref_sq)
            .and(&self.mu_ref)
            .for_each(|o, &crsq, &mr| *o = crsq - mr * mr);
        self.precomp_valid = true;
    }

    /// Invalidate precomputed ref values (call when ref_patch changes).
    pub fn invalidate_ref(&mut self) {
        self.precomp_valid = false;
    }
}

impl NeurogramSimiliarityIndexMeasure {
    /// Scalar similarity using pre-allocated scratch buffers.
    /// Avoids all heap allocations in the hot DP loop.
    ///
    /// If `s.precomp_valid` is true, skips recomputing reference-only values
    /// (mu_ref, conv_ref_sq, sigma_ref_sq), saving 2 out of 5 conv2d calls
    /// and 1 element-wise op per invocation.
    #[inline]
    pub fn measure_similarity_scalar_scratched(
        &self,
        ref_patch: &Array2<f64>,
        deg_patch: &Array2<f64>,
        s: &mut NsimScratch,
    ) -> f64 {
        let window = &*WINDOW;

        let c1 = (0.01 * self.intensity_range) * (0.01 * self.intensity_range);
        let c3 = (0.03 * self.intensity_range) * (0.03 * self.intensity_range) / 2.0;

        if !s.precomp_valid {
            // Compute reference-only values (fallback when not precomputed)
            conv2d_3x3_reflected_into(window, ref_patch, &mut s.mu_ref);
            Zip::from(&mut s.ref_neuro_sq)
                .and(ref_patch)
                .for_each(|o, &r| *o = r * r);
            conv2d_3x3_reflected_into(window, &s.ref_neuro_sq, &mut s.conv_ref_sq);
            Zip::from(&mut s.precomp_srs)
                .and(&s.conv_ref_sq)
                .and(&s.mu_ref)
                .for_each(|o, &crsq, &mr| *o = crsq - mr * mr);
        }

        // Degraded-only values (always recomputed)
        conv2d_3x3_reflected_into(window, deg_patch, &mut s.mu_deg);
        Zip::from(&mut s.deg_neuro_sq)
            .and(deg_patch)
            .for_each(|o, &d| *o = d * d);
        Zip::from(&mut s.ref_neuro_deg)
            .and(ref_patch)
            .and(deg_patch)
            .for_each(|o, &r, &d| *o = r * d);
        conv2d_3x3_reflected_into(window, &s.deg_neuro_sq, &mut s.conv_deg_sq);
        conv2d_3x3_reflected_into(window, &s.ref_neuro_deg, &mut s.conv_rd);

        // Fused scalar accumulation using precomputed sigma_ref_squared
        let n = s.mu_ref.len() as f64;
        let mut sum = 0.0f64;
        Zip::from(&s.mu_ref)
            .and(&s.mu_deg)
            .and(&s.precomp_srs)
            .and(&s.conv_deg_sq)
            .and(&s.conv_rd)
            .for_each(|&mr, &md, &srs, &cdsq, &crd| {
                let dms = md * md;
                let mrd = mr * md;
                let rms = mr * mr;
                let sds = cdsq - dms;
                let srd = crd - mrd;
                let intensity = (2.0 * mrd + c1) / (rms + dms + c1);
                let structure_denom = {
                    let prod = srs * sds;
                    if prod < 0.0 { c3 } else { prod.sqrt() + c3 }
                };
                let structure = (srd + c3) / structure_denom;
                sum += intensity * structure;
            });
        sum / n
    }

    /// Full patch similarity using pre-allocated scratch buffers.
    /// Avoids the ~15 temporary Array2 allocations of `measure_patch_similarity`.
    pub fn measure_patch_similarity_scratched(
        &self,
        ref_patch: &Array2<f64>,
        deg_patch: &Array2<f64>,
        s: &mut NsimScratch,
    ) -> PatchSimilarityResult {
        let window = &*WINDOW;
        let nrows = ref_patch.nrows();
        let ncols = ref_patch.ncols();

        let c1 = (0.01 * self.intensity_range) * (0.01 * self.intensity_range);
        let c3 = (0.03 * self.intensity_range) * (0.03 * self.intensity_range) / 2.0;

        // Conv2d into scratch buffers
        conv2d_3x3_reflected_into(window, ref_patch, &mut s.mu_ref);
        conv2d_3x3_reflected_into(window, deg_patch, &mut s.mu_deg);

        // Compute squared inputs into scratch
        Zip::from(&mut s.ref_neuro_sq)
            .and(ref_patch)
            .for_each(|o, &r| *o = r * r);
        Zip::from(&mut s.deg_neuro_sq)
            .and(deg_patch)
            .for_each(|o, &d| *o = d * d);
        Zip::from(&mut s.ref_neuro_deg)
            .and(ref_patch)
            .and(deg_patch)
            .for_each(|o, &r, &d| *o = r * d);

        // Conv of squared/cross inputs
        conv2d_3x3_reflected_into(window, &s.ref_neuro_sq, &mut s.conv_ref_sq);
        conv2d_3x3_reflected_into(window, &s.deg_neuro_sq, &mut s.conv_deg_sq);
        conv2d_3x3_reflected_into(window, &s.ref_neuro_deg, &mut s.conv_rd);

        // Compute per-band statistics from the fused sim_map.
        // sim_map[r,c] = intensity * structure, computed element-wise.
        let inv_ncols = 1.0 / ncols as f64;
        let mut freq_band_means = vec![0.0f64; nrows];
        let mut freq_band_deg_energy = vec![0.0f64; nrows];
        // freq_band_stddevs computed in second pass below

        // Compute deg_patch row means
        for r in 0..nrows {
            let mut deg_sum = 0.0f64;
            for c in 0..ncols {
                deg_sum += deg_patch[(r, c)];
            }
            freq_band_deg_energy[r] = deg_sum * inv_ncols;
        }

        // Compute sim_map row means
        for r in 0..nrows {
            let mut sim_sum = 0.0f64;
            for c in 0..ncols {
                let mr = s.mu_ref[(r, c)];
                let md = s.mu_deg[(r, c)];
                let rms = mr * mr;
                let dms = md * md;
                let mrd = mr * md;
                let srs = s.conv_ref_sq[(r, c)] - rms;
                let sds = s.conv_deg_sq[(r, c)] - dms;
                let srd = s.conv_rd[(r, c)] - mrd;
                let intensity = (2.0 * mrd + c1) / (rms + dms + c1);
                let structure_denom = {
                    let prod = srs * sds;
                    if prod < 0.0 { c3 } else { prod.sqrt() + c3 }
                };
                let structure = (srd + c3) / structure_denom;
                let sim = intensity * structure;
                sim_sum += sim;

                // Accumulate for variance (two-pass: first pass = mean)
                // We store sim values temporarily in conv_rd (reused as scratch)
                s.conv_rd[(r, c)] = sim;
            }
            freq_band_means[r] = sim_sum * inv_ncols;
        }

        // Second pass for stddev
        let mut freq_band_stddevs = vec![0.0f64; nrows];
        if ncols > 1 {
            for r in 0..nrows {
                let mean = freq_band_means[r];
                let mut ss = 0.0f64;
                for c in 0..ncols {
                    let diff = s.conv_rd[(r, c)] - mean;
                    ss += diff * diff;
                }
                freq_band_stddevs[r] = (ss / (ncols as f64 - 1.0)).sqrt();
            }
        }

        let mean_freq_band_means = freq_band_means.iter().sum::<f64>() / nrows as f64;

        PatchSimilarityResult::new(
            freq_band_means,
            freq_band_stddevs,
            freq_band_deg_energy,
            mean_freq_band_means,
        )
    }
}

impl PatchSimilarityComparator for NeurogramSimiliarityIndexMeasure {
    /// Computes the NSIM between `ref_patch` and `deg_patch` and returns the mean and standard deviation of each frequency band, the energy of the degraded patch and the similarity score.
    fn measure_patch_similarity(
        &self,
        ref_patch: &ndarray::Array2<f64>,
        deg_patch: &ndarray::Array2<f64>,
    ) -> PatchSimilarityResult {
        let window = &*WINDOW;

        let k = [0.01, 0.03];
        let c1 = (k[0] * self.intensity_range).powf(2.0);
        let c3 = (k[1] * self.intensity_range).powf(2.0) / 2.0;

        // Compute mu
        let mu_ref = perform_valid_2d_conv_with_boundary(&window, ref_patch);
        let mu_deg = perform_valid_2d_conv_with_boundary(&window, deg_patch);

        let ref_mu_squared = &mu_ref * &mu_ref;
        let deg_mu_squared = &mu_deg * &mu_deg;
        let mu_r_mu_d = &mu_ref * &mu_deg;

        // Compute squared arrays without cloning (mapv allocates one new array each)
        let ref_neuro_sq = ref_patch.mapv(|x| x * x);
        let deg_neuro_sq = deg_patch.mapv(|x| x * x);

        // Compute sigmas
        let conv2_ref_neuro_squared =
            perform_valid_2d_conv_with_boundary(&window, &ref_neuro_sq);
        let sigma_ref_squared = &conv2_ref_neuro_squared - &ref_mu_squared;

        let conv2_deg_neuro_squared =
            perform_valid_2d_conv_with_boundary(&window, &deg_neuro_sq);
        let sigma_deg_squared = &conv2_deg_neuro_squared - &deg_mu_squared;

        // Compute cross-product without cloning
        let ref_neuro_deg = Zip::from(&*ref_patch)
            .and(&*deg_patch)
            .map_collect(|&r, &d| r * d);
        let conv2_ref_neuro_deg = perform_valid_2d_conv_with_boundary(&window, &ref_neuro_deg);

        let sigma_r_d = &conv2_ref_neuro_deg - &mu_r_mu_d;

        // Compute intensity
        let intensity_numerator = &mu_r_mu_d * 2.0 + c1;
        let intensity_denominator = &ref_mu_squared + &deg_mu_squared + c1;

        let intensity = &intensity_numerator / &intensity_denominator;

        // Compute structure
        let structure_numerator = &sigma_r_d + c3;

        // Compute structure denominator: sqrt(sigma_ref^2 * sigma_deg^2) + c3
        // Fused into a single allocation with map_collect
        let structure_denominator = Zip::from(&sigma_ref_squared)
            .and(&sigma_deg_squared)
            .map_collect(|&sr, &sd| {
                let prod = sr * sd;
                if prod < 0.0 { c3 } else { prod.sqrt() + c3 }
            });

        let structure = &structure_numerator / &structure_denominator;
        let sim_map = &intensity * &structure;

        let freq_band_deg_energy: Array1<f64> = deg_patch
            .mean_axis(Axis(1))
            .expect("Failed to compute mean for degraded signal!");
        let freq_band_means: Array1<f64> = sim_map
            .mean_axis(Axis(1))
            .expect("Failed to compute mean for similarity map!");
        let freq_band_std: Array1<f64> = sim_map.std_axis(Axis(1), 1.0);
        let mean_freq_band_means = freq_band_means
            .mean()
            .expect("Failed to compute mean of means for degraded signal!");

        PatchSimilarityResult::new(
            freq_band_means.to_vec(),
            freq_band_std.to_vec(),
            freq_band_deg_energy.to_vec(),
            mean_freq_band_means,
        )
    }

    fn measure_similarity_scalar(
        &self,
        ref_patch: &ndarray::Array2<f64>,
        deg_patch: &ndarray::Array2<f64>,
    ) -> f64 {
        let window = &*WINDOW;

        let k = [0.01, 0.03];
        let c1 = (k[0] * self.intensity_range).powf(2.0);
        let c3 = (k[1] * self.intensity_range).powf(2.0) / 2.0;

        let mu_ref = perform_valid_2d_conv_with_boundary(window, ref_patch);
        let mu_deg = perform_valid_2d_conv_with_boundary(window, deg_patch);

        let ref_mu_squared = &mu_ref * &mu_ref;
        let deg_mu_squared = &mu_deg * &mu_deg;
        let mu_r_mu_d = &mu_ref * &mu_deg;

        let ref_neuro_sq = ref_patch.mapv(|x| x * x);
        let deg_neuro_sq = deg_patch.mapv(|x| x * x);

        let conv2_ref_neuro_squared =
            perform_valid_2d_conv_with_boundary(window, &ref_neuro_sq);
        let sigma_ref_squared = &conv2_ref_neuro_squared - &ref_mu_squared;

        let conv2_deg_neuro_squared =
            perform_valid_2d_conv_with_boundary(window, &deg_neuro_sq);
        let sigma_deg_squared = &conv2_deg_neuro_squared - &deg_mu_squared;

        let ref_neuro_deg = Zip::from(&*ref_patch)
            .and(&*deg_patch)
            .map_collect(|&r, &d| r * d);
        let conv2_ref_neuro_deg =
            perform_valid_2d_conv_with_boundary(window, &ref_neuro_deg);

        let sigma_r_d = &conv2_ref_neuro_deg - &mu_r_mu_d;

        // Compute scalar mean of (intensity * structure) without materializing
        // per-band statistics or allocating Vec results.
        let n = mu_r_mu_d.len() as f64;
        let mut sum = 0.0f64;
        Zip::from(&mu_r_mu_d)
            .and(&ref_mu_squared)
            .and(&deg_mu_squared)
            .and(&sigma_r_d)
            .and(&sigma_ref_squared)
            .and(&sigma_deg_squared)
            .for_each(|&mrd, &rms, &dms, &srd, &srs, &sds| {
                let intensity = (2.0 * mrd + c1) / (rms + dms + c1);
                let structure_denom = {
                    let prod = srs * sds;
                    if prod < 0.0 { c3 } else { prod.sqrt() + c3 }
                };
                let structure = (srd + c3) / structure_denom;
                sum += intensity * structure;
            });
        sum / n
    }
}

#[cfg(test)]
mod tests {

    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    use super::*;

    #[test]
    fn test_neurogram_measure() {
        let ref_patch = vec![1.0, 0.0, 0.0];
        let ref_patch_mat = Array2::from_shape_vec((3, 1), ref_patch).unwrap();
        let deg_patch = vec![0.0, 0.0, 0.0];
        let deg_patch_mat = Array2::from_shape_vec((3, 1), deg_patch).unwrap();
        let expected_result = [0.000125225, 0.00875062, 1.0];

        let sim_comparator = NeurogramSimiliarityIndexMeasure::default();

        let result =
            sim_comparator.measure_patch_similarity(&ref_patch_mat, &deg_patch_mat);

        assert_abs_diff_eq!(
            result.freq_band_means[0],
            expected_result[0],
            epsilon = 0.0001
        );
        assert_abs_diff_eq!(
            result.freq_band_means[1],
            expected_result[1],
            epsilon = 0.0001
        );
        assert_abs_diff_eq!(
            result.freq_band_means[2],
            expected_result[2],
            epsilon = 0.0001
        );
    }
}
