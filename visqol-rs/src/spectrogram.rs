use ndarray::Array2;
use ndarray_stats::QuantileExt;

/// Contains the spectral representation of audio data
#[derive(Clone)]
pub struct Spectrogram {
    /// Spectrogram data, rows signify center frequencies, columns signify time
    pub data: Array2<f64>,
    /// Center frequencies in Hz
    pub center_freq_bands: Vec<f64>,
}

impl Spectrogram {
    /// Creates a new spectrogram. Note that `data` and `center_freq_bands` are moved
    pub fn new(data: Array2<f64>, center_freq_bands: Vec<f64>) -> Self {
        Self {
            data,
            center_freq_bands,
        }
    }

    /// Converts the spectrogram from linear scale to dB scale
    pub fn convert_to_db(&mut self) {
        let sample_to_db = |element: f64| {
            let sample: f64 = if element == 0.0 {
                f64::EPSILON
            } else {
                element.abs()
            };
            10.0 * (sample.log10())
        };
        self.data.mapv_inplace(sample_to_db);
    }

    /// Converts to dB scale and clamps to floor in a single pass.
    pub fn convert_to_db_and_raise_floor(&mut self, floor: f64) {
        self.data.mapv_inplace(|element| {
            let sample: f64 = if element == 0.0 {
                f64::EPSILON
            } else {
                element.abs()
            };
            let db = 10.0 * sample.log10();
            if db > floor { db } else { floor }
        });
    }

    /// Returns the minimum value of the spectrogram
    pub fn get_minimum(&self) -> f64 {
        *self
            .data
            .min()
            .expect("Failed to compute minimum for spectrogram")
    }

    /// Returns the minimum and subtracts it from both spectrograms in one pass
    pub fn get_minimum_and_subtract_floor_pair(a: &mut Self, b: &mut Self) {
        let a_min = *a.data.min().expect("Failed to compute minimum");
        let b_min = *b.data.min().expect("Failed to compute minimum");
        let lowest = a_min.min(b_min);
        a.data -= lowest;
        b.data -= lowest;
    }

    /// Elementwise subtraction of the spectrogram
    pub fn subtract_floor(&mut self, floor: f64) { self.data -= floor; }

    /// Clamps each value in the spectrogram to `new_floor`
    pub fn raise_floor(&mut self, new_floor: f64) {
        self.data.mapv_inplace(|element| new_floor.max(element));
    }

    /// Given a noise threshold and a second spectrogram, both spectrograms are raised to share the same noise floor specified by `noise_threshold`
    pub fn raise_floor_per_frame(&mut self, noise_threshold: f64, other: &mut Self) {
        let nrows = self.data.nrows();
        let ncols_self = self.data.ncols();
        let ncols_other = other.data.ncols();
        let min_columns = ncols_self.min(ncols_other);

        // Work on raw slices for cache-friendly strided access.
        let self_slice = self.data.as_slice_mut()
            .expect("spectrogram data not contiguous");
        let other_slice = other.data.as_slice_mut()
            .expect("spectrogram data not contiguous");

        for col in 0..min_columns {
            // Find max across both spectrograms for this column.
            // Data is row-major: element (r, c) is at offset r * ncols + c.
            let mut any_max = f64::NEG_INFINITY;
            for r in 0..nrows {
                let sv = self_slice[r * ncols_self + col];
                let ov = other_slice[r * ncols_other + col];
                let m = sv.max(ov);
                if m > any_max { any_max = m; }
            }
            let floor_db = any_max - noise_threshold;

            // Clamp both columns to floor
            for r in 0..nrows {
                let si = r * ncols_self + col;
                if self_slice[si] < floor_db { self_slice[si] = floor_db; }
                let oi = r * ncols_other + col;
                if other_slice[oi] < floor_db { other_slice[oi] = floor_db; }
            }
        }
    }
}

impl std::ops::Index<(usize, usize)> for Spectrogram {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output { &self.data[index] }
}

impl std::ops::IndexMut<(usize, usize)> for Spectrogram {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output { &mut self.data[index] }
}

#[cfg(test)]
mod tests {
    use crate::test_utility;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    use super::*;

    const TOLERANCE: f64 = 0.0001;
    const MIN_ELEM: f64 = -53.2;
    const FLOOR: f64 = 0.1;

    #[test]
    fn convert_to_db_test() {
        let elements = Array2::<f64>::from_shape_vec(
            (10, 1),
            vec![
                10.21, -4.63, 0.54, 87.98, 0.065, 0.0, MIN_ELEM, 8.7, 0.0, -2.76,
            ],
        )
        .unwrap();

        let elements_db_scaled = Array2::<f64>::from_shape_vec(
            (10, 1),
            vec![
                10.0903, 6.6558, -2.6761, 19.4438, -11.8709, -156.5356, 17.2591, 9.3952, -156.5356,
                4.4091,
            ],
        )
        .unwrap();

        let mut spectrogram = Spectrogram::new(elements, vec![]);
        spectrogram.convert_to_db();

        test_utility::compare_real_matrix(&spectrogram.data, &elements_db_scaled, TOLERANCE);
    }

    #[test]
    fn minimum_test() {
        let elements = Array2::<f64>::from_shape_vec(
            (10, 1),
            vec![
                10.21, -4.63, 0.54, 87.98, 0.065, 0.0, MIN_ELEM, 8.7, 0.0, -2.76,
            ],
        )
        .unwrap();

        let spectrogram = Spectrogram::new(elements, vec![]);

        assert_abs_diff_eq!(spectrogram.get_minimum(), MIN_ELEM, epsilon = TOLERANCE);
    }

    #[test]
    fn subtract_floor_test() {
        let elements = Array2::<f64>::from_shape_vec(
            (10, 1),
            vec![
                10.21, -4.63, 0.54, 87.98, 0.065, 0.0, MIN_ELEM, 8.7, 0.0, -2.76,
            ],
        )
        .unwrap();

        let elements_floor_subtracted = Array2::<f64>::from_shape_vec(
            (10, 1),
            vec![
                10.21 - FLOOR,
                -4.63 - FLOOR,
                0.54 - FLOOR,
                87.98 - FLOOR,
                0.065 - FLOOR,
                0.0 - FLOOR,
                MIN_ELEM - FLOOR,
                8.7 - FLOOR,
                0.0 - FLOOR,
                -2.76 - FLOOR,
            ],
        )
        .unwrap();

        let mut spectrogram = Spectrogram::new(elements, vec![]);

        spectrogram.subtract_floor(FLOOR);
        test_utility::compare_real_matrix(&spectrogram.data, &elements_floor_subtracted, TOLERANCE);
    }
}
