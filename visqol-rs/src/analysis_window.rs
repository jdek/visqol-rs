/// Temporal analysis window used for creating spectrograms
pub struct AnalysisWindow {
    /// Size of the window in samples
    pub size: usize,
    /// Overlap of the window in milliseconds
    pub overlap: f64,
    /// Precomputed Hann window coefficients, length = size
    pub hann: Vec<f64>,
}

impl AnalysisWindow {
    /// Creates a new analysis window based on sample rate, desired overlap and duration
    pub fn new(sample_rate: u32, overlap: f64, window_duration: f64) -> Self {
        let size = (sample_rate as f64 * window_duration).round() as usize;
        // Hann: 0.5 - 0.5 * cos(2π i / (size - 1)), matching C++ visqol.
        let denom = (size as f64 - 1.0).max(1.0);
        let hann = (0..size)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / denom).cos())
            .collect();
        Self {
            size,
            overlap,
            hann,
        }
    }
}
