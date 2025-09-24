use crate::similarity_to_quality_mapper::SimilarityToQualityMapper;
use crate::support_vector_regression_model::SupportVectorRegressionModel;

/// Maps a a similarity score to a MOS using support vector regression.
pub struct SvrSimilarityToQualityMapper {
    model: SupportVectorRegressionModel,
}

impl SvrSimilarityToQualityMapper {
    /// Initializes the mapper with a pre-constructed `SupportVectorRegressionModel`.
    pub fn new(model: SupportVectorRegressionModel) -> Self {
        Self { model }
    }
}

impl SimilarityToQualityMapper for SvrSimilarityToQualityMapper {
    fn predict_quality(&self, similarity_vector: &[f64]) -> f64 {
        let solution = self.model.predict(similarity_vector);
        solution.clamp(1.0, 5.0)
    }
}
