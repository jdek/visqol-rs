use std::error::Error;

use crate::alignment::align_and_truncate;
use crate::constants;
use crate::gammatone_filterbank::GammatoneFilterbank;
use crate::gammatone_spectrogram_builder::GammatoneSpectrogramBuilder;
use crate::{
    analysis_window::AnalysisWindow,
    audio_signal::AudioSignal,
    audio_utils,
    neurogram_similiarity_index_measure::{NeurogramSimiliarityIndexMeasure, NsimScratch},
    patch_similarity_comparator::PatchSimilarityResult,
    spectrogram_builder::SpectrogramBuilder,
    visqol_error::VisqolError,
};
use ndarray::{concatenate, s, Array1, Array2, Axis};
pub struct ComparisonPatchesSelector {
    sim_comparator: NeurogramSimiliarityIndexMeasure,
}

impl ComparisonPatchesSelector {
    pub fn new(sim_comparator: NeurogramSimiliarityIndexMeasure) -> Self { Self { sim_comparator } }

    /// This function composes the most suitable patches in a degraded signal given a reference signal.
    pub fn find_most_optimal_deg_patches(
        &self,
        ref_patches: &mut [Array2<f64>],
        ref_patch_indices: &mut [usize],
        spectrogram_data: &Array2<f64>,
        frame_duration: f64,
        search_window_radius: i32,
    ) -> Result<Vec<PatchSimilarityResult>, VisqolError> {
        let num_frames_per_patch = ref_patches[0].ncols();
        let num_frames_in_deg_spectro = spectrogram_data.ncols();
        let patch_duration = frame_duration * num_frames_per_patch as f64;
        let search_window = search_window_radius * num_frames_per_patch as i32;
        let num_patches = Self::calc_max_num_patches(
            ref_patch_indices,
            num_frames_in_deg_spectro,
            num_frames_per_patch,
        );

        if num_patches == 0 {
            return Err(VisqolError::SignalsTooDifferent);
        } else if num_patches < ref_patch_indices.len() {
            log::warn!(
                "Warning: Dropping {} (of {}) reference patches
            due to the degraded file being misaligned or too short. If too many
            patches are dropped, the score will be less meaningful.",
                ref_patch_indices.len() - num_patches,
                ref_patch_indices.len()
            );
        }

        // The vector to store the similarity results
        let mut best_deg_patches = Vec::<PatchSimilarityResult>::new();
        best_deg_patches.resize(num_patches, PatchSimilarityResult::default());

        let dp_cols = spectrogram_data.ncols();
        let dp_rows = ref_patch_indices.len();
        let mut cumulative_similarity_dp = vec![0.0f64; dp_rows * dp_cols];
        let mut backtrace = vec![0usize; dp_rows * dp_cols];

        // Only build deg patches within the search window range to avoid
        // allocating patches for offsets that will never be visited.
        let global_lower = if ref_patch_indices.is_empty() {
            0
        } else {
            (ref_patch_indices[0] as i32 - search_window).max(0) as usize
        };
        let global_upper = if ref_patch_indices.is_empty() {
            0
        } else {
            (*ref_patch_indices.last().unwrap() as i32 + search_window + 1)
                .min(spectrogram_data.ncols() as i32) as usize
        };

        let mut deg_patches = Vec::<Array2<f64>>::with_capacity(spectrogram_data.ncols());

        for slide_offset in 0..spectrogram_data.ncols() {
            if slide_offset >= global_lower && slide_offset < global_upper {
                deg_patches.push(Self::build_degraded_patch(
                    spectrogram_data,
                    slide_offset,
                    slide_offset + ref_patches[0].ncols(),
                ));
            } else {
                // Placeholder - won't be accessed
                deg_patches.push(Array2::zeros((0, 0)));
            }
        }

        // Pre-allocate scratch buffers for the NSIM scalar computation.
        let mut scratch = if !ref_patches.is_empty() {
            NsimScratch::new(ref_patches[0].nrows(), ref_patches[0].ncols())
        } else {
            NsimScratch::new(0, 0)
        };

        // Attempt to get a good alignment with backtracking.
        for (index, ref_patch) in ref_patches.iter_mut().enumerate() {
            // Precompute reference-only conv2d values once per ref_patch,
            // saving 2 out of 5 conv2d calls per DP search offset.
            scratch.precompute_ref(ref_patch);
            self.find_most_optimal_deg_patch(
                spectrogram_data,
                ref_patch,
                &mut deg_patches,
                &mut cumulative_similarity_dp,
                &mut backtrace,
                ref_patch_indices,
                index,
                search_window,
                dp_cols,
                &mut scratch,
            );
        }
        let mut max_similarity_score = f64::MIN;
        // The patch index for the last reference patch.
        let last_index = num_patches - 1;

        // The last_offset stores the offset at which the last reference patch got the
        // maximal similarity score over all the reference patches.

        let mut last_offset = 0;

        let lower_limit = 0.max(ref_patch_indices[last_index] as i32 - search_window) as usize;

        // The for loop is used to find the offset which maximizes the similarity
        // score across all the patches.
        // +1 for including last
        for slide_offset in lower_limit..ref_patch_indices[last_index] + search_window as usize + 1
        {
            if slide_offset >= num_frames_in_deg_spectro {
                // The frame offset for degraded start patch cannot be more than the
                // number of frames in the degraded spectrogram.
                break;
            }

            if cumulative_similarity_dp[last_index * dp_cols + slide_offset] > max_similarity_score {
                max_similarity_score = cumulative_similarity_dp[last_index * dp_cols + slide_offset];
                last_offset = slide_offset;
            }
        }

        scratch.invalidate_ref();
        let mut patch_index: i32 = (num_patches - 1) as i32;
        while patch_index >= 0 {
            let pi = patch_index as usize;
            let ref_ncols = ref_patches[pi].ncols();

            // Build a temporary deg patch only when outside the pre-built range.
            let tmp_deg_patch;
            let deg_patch_ref = if last_offset >= global_lower && last_offset < global_upper {
                &deg_patches[last_offset]
            } else {
                tmp_deg_patch = Self::build_degraded_patch(
                    spectrogram_data,
                    last_offset,
                    last_offset + ref_ncols,
                );
                &tmp_deg_patch
            };

            best_deg_patches[pi] = self
                .sim_comparator
                .measure_patch_similarity_scratched(&ref_patches[pi], deg_patch_ref, &mut scratch);

            // This condition is true only if no matching patch was found for the given
            // reference patch. In this case, the matched patch is essentially set to
            // NULL (which is different from a silent patch).

            if last_offset == backtrace[pi * dp_cols + last_offset] {
                best_deg_patches[pi].deg_patch_start_time = 0.0;
                best_deg_patches[pi].deg_patch_end_time = 0.0;
                best_deg_patches[pi].similarity = 0.0;
                let num_rows = best_deg_patches[pi].freq_band_means.len();
                best_deg_patches[pi].freq_band_means = vec![0.0; num_rows];
            } else {
                best_deg_patches[pi].deg_patch_start_time =
                    last_offset as f64 * frame_duration;
                best_deg_patches[pi].deg_patch_end_time =
                    best_deg_patches[pi].deg_patch_start_time + patch_duration;
            }

            best_deg_patches[pi].ref_patch_start_time =
                ref_patch_indices[pi] as f64 * frame_duration;
            best_deg_patches[pi].ref_patch_end_time =
                best_deg_patches[pi].ref_patch_start_time + patch_duration;
            last_offset = backtrace[pi * dp_cols + last_offset];

            patch_index -= 1;
        }
        Ok(best_deg_patches)
    }

    /// This function finds the most suitable patch in a degraded signal given a reference patch.
    pub fn find_most_optimal_deg_patch(
        &self,
        spectrogram_data: &Array2<f64>,
        ref_patch: &mut Array2<f64>,
        deg_patches: &mut [Array2<f64>],
        cumulative_similarity_dp: &mut [f64],
        backtrace: &mut [usize],
        ref_patch_indices: &[usize],
        patch_index: usize,
        search_window: i32,
        dp_cols: usize,
        scratch: &mut NsimScratch,
    ) {
        let ref_frame_index = ref_patch_indices[patch_index];

        let start_offset = (ref_frame_index as i32 - search_window).max(0);
        let end_offset = (ref_frame_index as i32 + search_window)
            .min(spectrogram_data.ncols() as i32 - 1);

        // For patch_index > 0: maintain a running max over
        // dp[prev_row][lower_limit..slide_offset] so we avoid rescanning
        // the previous row at every offset (O(1) instead of O(search_window)).
        let (prev_row, lower_limit) = if patch_index > 0 {
            let prev = (patch_index - 1) * dp_cols;
            let ll = (ref_patch_indices[patch_index - 1] as i32 - search_window).max(0);
            (prev, ll)
        } else {
            (0, 0)
        };
        let mut running_max = f64::MIN;
        let mut running_max_idx: i32 = -1;

        // Seed running_max with dp values from lower_limit up to (but not
        // including) start_offset — these are offsets that the original
        // backward scan would have covered on the very first iteration.
        if patch_index > 0 {
            for i in lower_limit..start_offset {
                let val = cumulative_similarity_dp[prev_row + i as usize];
                if val > running_max {
                    running_max = val;
                    running_max_idx = i;
                }
            }
        }

        for slide_offset in start_offset..=end_offset {
            let deg_patch = &deg_patches[slide_offset as usize];
            let mut similarity = self
                .sim_comparator
                .measure_similarity_scalar_scratched(ref_patch, deg_patch, scratch);
            let mut past_slide_offset: i32 = -1;

            if patch_index > 0 {
                // running_max holds max of dp[prev][lower_limit..slide_offset]
                let highest_sim = running_max;
                past_slide_offset = running_max_idx;

                similarity += highest_sim;

                // Check "no-move" option: previous patch at same offset
                if cumulative_similarity_dp[prev_row + slide_offset as usize]
                    > similarity
                {
                    similarity =
                        cumulative_similarity_dp[prev_row + slide_offset as usize];
                    past_slide_offset = slide_offset;
                }

                // Extend running_max to include current slide_offset for next iteration
                let cur_prev_val = cumulative_similarity_dp[prev_row + slide_offset as usize];
                if cur_prev_val > running_max {
                    running_max = cur_prev_val;
                    running_max_idx = slide_offset;
                }
            }
            cumulative_similarity_dp[patch_index * dp_cols + slide_offset as usize] = similarity;
            backtrace[patch_index * dp_cols + slide_offset as usize] = past_slide_offset as usize;
        }
    }

    /// Calculate the maximum number of patches that the degraded spectrogram can support.
    pub fn calc_max_num_patches(
        ref_patch_indices: &[usize],
        num_frames_in_deg_spectro: usize,
        num_frames_per_patch: usize,
    ) -> usize {
        let mut num_patches = ref_patch_indices.len();

        if num_patches != 0 {
            while (ref_patch_indices[num_patches - 1] - (num_frames_per_patch / 2))
                > num_frames_in_deg_spectro
            {
                num_patches -= 1;
            }
        }
        num_patches
    }

    /// Given an `AudioSignal` and the desired start and end times in seconds, this function returns a copy of the segment in the audio signal ranging from `start_time` to `end_time`
    pub fn slice(in_signal: &AudioSignal, start_time: f64, end_time: f64) -> AudioSignal {
        let start_index = ((start_time * in_signal.sample_rate as f64) as usize).max(0);
        let end_index =
            ((end_time * in_signal.sample_rate as f64) as usize).min(in_signal.data_matrix.len());

        let mut sliced_matrix = in_signal
            .data_matrix
            .slice(s![start_index..end_index])
            .to_owned();
        let end_time_diff =
            (end_time * in_signal.sample_rate as f64 - in_signal.data_matrix.len() as f64) as usize;

        if end_time_diff > 0 {
            let post_silence_matrix = Array1::<f64>::zeros(end_time_diff);
            sliced_matrix =
                concatenate(Axis(0), &[sliced_matrix.view(), post_silence_matrix.view()])
                    .expect("Failed to zero-pad patch!");
        }

        if start_time < 0.0 {
            let pre_silence_matrix =
                Array1::<f64>::zeros((-start_time * in_signal.sample_rate as f64) as usize);
            sliced_matrix =
                concatenate(Axis(0), &[pre_silence_matrix.view(), sliced_matrix.view()])
                    .expect("Failed to zero-pad patch!");
        }
        AudioSignal::from_owned(sliced_matrix, in_signal.sample_rate)
    }

    pub fn build_degraded_patch(
        spectrogram_data: &Array2<f64>,
        window_beginning: usize,
        window_end: usize,
    ) -> Array2<f64> {
        let first_real_frame = 0.max(window_beginning);
        let last_real_frame = window_end.min(spectrogram_data.ncols());

        let mut deg_patch = spectrogram_data
            .slice(s![.., first_real_frame..last_real_frame])
            .to_owned();

        if window_end > spectrogram_data.ncols() {
            let append_matrix = Array2::<f64>::zeros((
                spectrogram_data.nrows(),
                window_end - spectrogram_data.ncols(),
            ));

            deg_patch = concatenate(Axis(1), &[deg_patch.view(), append_matrix.view()])
                .expect("Could not zero-pad patch!");
        }
        deg_patch
    }

    /// Performs alignment on a per-patch level.
    pub fn finely_align_and_recreate_patches<const NUM_BANDS: usize>(
        &self,
        sim_results: &mut [PatchSimilarityResult],
        ref_signal: &AudioSignal,
        deg_signal: &AudioSignal,
        analysis_window: &AnalysisWindow,
    ) -> Result<Vec<PatchSimilarityResult>, Box<dyn Error>> {
        // Reuse a single spectrogram builder across all patches so that
        // filter coefficients are computed once and the internal buffers
        // are re-used.
        let mut spect_builder = GammatoneSpectrogramBuilder::<NUM_BANDS>::new(
            GammatoneFilterbank::new(constants::MINIMUM_FREQ),
        );

        // Case: The patches are already matched.  Iterate over each pair.
        let mut realigned_results = Vec::<PatchSimilarityResult>::with_capacity(sim_results.len());
        realigned_results.resize(sim_results.len(), PatchSimilarityResult::default());
        let mut scratch = NsimScratch::new(0, 0);
        for (i, result) in sim_results.iter_mut().enumerate() {
            if result.deg_patch_start_time == result.deg_patch_end_time
                && result.deg_patch_start_time == 0.0
            {
                realigned_results[i] = result.clone();
                continue;
            }
            // 1. The sim results keep track of the start and end points of each matched
            // pair.  Extract the audio for this segment.
            let ref_patch_audio = Self::slice(
                ref_signal,
                result.ref_patch_start_time,
                result.ref_patch_end_time,
            );
            let deg_patch_audio = Self::slice(
                deg_signal,
                result.deg_patch_start_time,
                result.deg_patch_end_time,
            );

            // 2. For any pair, we want to shift the degraded signal to be maximally
            // aligned.
            let (ref_audio_aligned, deg_audio_aligned, lag) =
                align_and_truncate(&ref_patch_audio, &deg_patch_audio)
                    .ok_or(VisqolError::FailedToAlignSignals)?;

            let new_ref_duration = ref_audio_aligned.get_duration();
            let new_deg_duration = deg_audio_aligned.get_duration();
            // 3. Compute a new spectrogram for the degraded audio.
            let mut ref_spectrogram = spect_builder.build(&ref_audio_aligned, analysis_window)?;
            let mut deg_spectrogram = spect_builder.build(&deg_audio_aligned, analysis_window)?;
            // 4. Recreate an aligned degraded patch from the new spectrogram.

            audio_utils::prepare_spectrograms_for_comparison(
                &mut ref_spectrogram,
                &mut deg_spectrogram,
            );
            // 5. Update the similarity result with the new patch.

            scratch.ensure_size(ref_spectrogram.data.nrows(), ref_spectrogram.data.ncols());
            let mut new_sim_result = self
                .sim_comparator
                .measure_patch_similarity_scratched(&ref_spectrogram.data, &deg_spectrogram.data, &mut scratch);
            // Compare to the old result and take the max.
            if new_sim_result.similarity < result.similarity {
                realigned_results[i] = result.clone();
            } else {
                if lag > 0.0 {
                    new_sim_result.ref_patch_start_time = result.ref_patch_start_time + lag;
                    new_sim_result.deg_patch_start_time = result.deg_patch_start_time;
                } else {
                    new_sim_result.ref_patch_start_time = result.ref_patch_start_time;
                    new_sim_result.deg_patch_start_time = result.deg_patch_start_time - lag;
                }
                new_sim_result.ref_patch_end_time =
                    new_sim_result.ref_patch_start_time + new_ref_duration;
                new_sim_result.deg_patch_end_time =
                    new_sim_result.deg_patch_start_time + new_deg_duration;
                realigned_results[i] = new_sim_result;
            }
        }
        Ok(realigned_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        audio_signal::AudioSignal, image_patch_creator::ImagePatchCreator,
        neurogram_similiarity_index_measure::NeurogramSimiliarityIndexMeasure,
        patch_creator::PatchCreator,
    };
    use ndarray::{arr2, Array1, Array2};

    #[test]
    fn num_patches_is_computed_correctly() {
        let patch_indices = vec![0, 15, 30, 45, 60];

        let slide_offset = 45;
        let accepted_num_patches =
            ComparisonPatchesSelector::calc_max_num_patches(&patch_indices, slide_offset, 30);

        assert_eq!(patch_indices.len(), accepted_num_patches);

        let slide_offset = 44;
        let accepted_num_patches =
            ComparisonPatchesSelector::calc_max_num_patches(&patch_indices, slide_offset, 30);

        assert_eq!(patch_indices.len() - 1, accepted_num_patches);
    }

    #[test]
    fn time_slicing_signal_is_sample_accurate() {
        let fs = 16000;
        let num_seconds = 3;

        let mut silence_matrix = Array1::zeros(fs * num_seconds);

        silence_matrix[16000] = 1.0;
        let three_seconds_silence = AudioSignal::new(silence_matrix.as_slice().unwrap(), fs as u32);

        let sliced_signal = ComparisonPatchesSelector::slice(&three_seconds_silence, 0.5, 2.5);

        assert_eq!(sliced_signal.get_duration(), 2.0);
        assert_eq!(sliced_signal[7999], 0.0);
        assert_eq!(sliced_signal[8000], 1.0);
        assert_eq!(sliced_signal[8001], 0.0);
    }

    #[test]
    fn optimal_patches_start_times_are_correct() {
        let ref_matrix = arr2(&[
            [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0],
            [0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0],
            [0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0],
        ]);

        // Create reference patches from given patch indices
        let patch_size = 1;
        let mut patch_indices: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        let patch_creator = ImagePatchCreator::new(patch_size);
        let mut ref_patches =
            patch_creator.create_patches_from_indices(&ref_matrix, &patch_indices);

        // Defining the degraded audio matrix
        let rows_concatenated: Vec<f64> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 3.0, 2.0,
            0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(rows_concatenated.len(), 3 * 30);
        let deg_matrix = Array2::from_shape_vec((3, 30), rows_concatenated).unwrap();

        let frame_duration = 1.0;
        let search_window = 8;

        let sim_measurer = NeurogramSimiliarityIndexMeasure::default();
        let selector = ComparisonPatchesSelector::new(sim_measurer);

        let res = selector
            .find_most_optimal_deg_patches(
                &mut ref_patches,
                &mut patch_indices,
                &deg_matrix,
                frame_duration,
                search_window,
            )
            .unwrap();

        assert_eq!(res[3].deg_patch_start_time, 0.0);
        assert_eq!(res[4].deg_patch_start_time, 7.0);
        assert_eq!(res[5].deg_patch_start_time, 8.0);
    }

    #[test]
    fn matches_are_out_of_order() {
        let ref_matrix = arr2(&[[1.0, 100.0, 3.0, 4.0], [0.0; 4], [1.0, 100.0, 3.0, 4.0]]);

        let patch_size = 1;
        let mut patch_indices = vec![0, 1, 2, 3];

        let patch_creator = ImagePatchCreator::new(patch_size);
        let mut ref_patches =
            patch_creator.create_patches_from_indices(&ref_matrix, &patch_indices);

        let deg_matrix = arr2(&[[100.0, 1.0, 3.0, 4.0], [0.0; 4], [100.0, 1.0, 3.0, 4.0]]);

        let frame_duration = 1.0;
        let search_window = 60;

        let sim_measurer = NeurogramSimiliarityIndexMeasure::default();
        let selector = ComparisonPatchesSelector::new(sim_measurer);

        let res = selector
            .find_most_optimal_deg_patches(
                &mut ref_patches,
                &mut patch_indices,
                &deg_matrix,
                frame_duration,
                search_window,
            )
            .unwrap();

        assert_eq!(res[0].deg_patch_start_time, 1.0);
        assert_eq!(res[1].deg_patch_start_time, 0.0);
        assert_eq!(res[2].deg_patch_start_time, 2.0);
        assert_eq!(res[3].deg_patch_start_time, 3.0);
    }

    #[test]
    fn results_are_different() {
        let ref_matrix = arr2(&[[1.0], [1.0], [0.0]]);

        let patch_size = 1;
        let mut patch_indices = vec![0];

        let patch_creator = ImagePatchCreator::new(patch_size);
        let mut ref_patches =
            patch_creator.create_patches_from_indices(&ref_matrix, &patch_indices);

        let concatenated_deg_mat = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let deg_matrix = Array2::from_shape_vec((3, 17), concatenated_deg_mat).unwrap();

        let frame_duration = 1.0;
        let search_window = 60;

        let sim_measurer = NeurogramSimiliarityIndexMeasure::default();
        let selector = ComparisonPatchesSelector::new(sim_measurer);

        let res = selector
            .find_most_optimal_deg_patches(
                &mut ref_patches,
                &mut patch_indices,
                &deg_matrix,
                frame_duration,
                search_window,
            )
            .unwrap();
        assert_eq!(res[0].deg_patch_start_time, 6.0);
    }

    #[test]
    fn start_times_in_longer_file_are_correct() {
        let ref_vec = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let ref_matrix = Array2::from_shape_vec((3, 31), ref_vec).unwrap();

        let patch_size = 2;

        let mut patch_indices = vec![4, 6, 10, 12, 14, 22];

        let patch_creator = ImagePatchCreator::new(patch_size);
        let mut ref_patches =
            patch_creator.create_patches_from_indices(&ref_matrix, &patch_indices);

        let deg_vec = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 2.0, 3.0,
            2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let deg_matrix = Array2::from_shape_vec((3, 31), deg_vec).unwrap();

        let frame_duration = 1.0;
        let search_window = 60;

        let sim_measurer = NeurogramSimiliarityIndexMeasure::default();
        let selector = ComparisonPatchesSelector::new(sim_measurer);

        let res = selector
            .find_most_optimal_deg_patches(
                &mut ref_patches,
                &mut patch_indices,
                &deg_matrix,
                frame_duration,
                search_window,
            )
            .unwrap();

        assert_eq!(res[0].deg_patch_start_time, 6.0);
        assert_eq!(res[1].deg_patch_start_time, 8.0);
        assert_eq!(res[2].deg_patch_start_time, 12.0);
        assert_eq!(res[3].deg_patch_start_time, 14.0);
        assert_eq!(res[4].deg_patch_start_time, 16.0);
        assert_eq!(res[5].deg_patch_start_time, 22.0);
    }
}
