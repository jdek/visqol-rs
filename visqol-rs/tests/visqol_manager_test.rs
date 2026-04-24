use approx::assert_abs_diff_eq;
use more_asserts::assert_gt;
use visqol_rs::{
    constants::{NUM_BANDS_AUDIO, NUM_BANDS_SPEECH},
    support_vector_regression_model::SupportVectorRegressionModel,
    variant::Variant,
    visqol_manager::VisqolManager,
};

const SEARCH_WINDOW: usize = 60;
const SVR_MODEL_PATH: &str = "../model/libsvm_nu_svr_model.txt";

fn fullband_manager() -> VisqolManager<NUM_BANDS_AUDIO> {
    let model = SupportVectorRegressionModel::new(SVR_MODEL_PATH);
    VisqolManager::new(Variant::Fullband { model }, SEARCH_WINDOW)
}

fn wideband_manager(use_unscaled_mos_mapping: bool) -> VisqolManager<NUM_BANDS_SPEECH> {
    VisqolManager::new(
        Variant::Wideband {
            use_unscaled_mos_mapping,
        },
        SEARCH_WINDOW,
    )
}

#[test]
fn regression_test_mono() {
    let (ref_path, deg_path) = speech_files();
    let result = fullband_manager().run(ref_path, deg_path).unwrap();
    // C++ reference: 1.76584 (visqol_cli --use_lattice_model=false)
    assert_abs_diff_eq!(result.moslqo, 1.76584, epsilon = 0.001);
}

#[test]
fn regression_test_stereo() {
    let (ref_path, deg_path) = guitar_files();
    let result = fullband_manager().run(ref_path, deg_path).unwrap();
    // C++ reference: 4.34972 (visqol_cli --use_lattice_model=false)
    assert_abs_diff_eq!(result.moslqo, 4.34972, epsilon = 0.001);
}

#[test]
fn test_identical_stddev_nsim() {
    let (ref_path, _) = guitar_files();
    let result = fullband_manager().run(ref_path, ref_path).unwrap();

    for nsim in result.fvnsim {
        assert_eq!(nsim, 1.0);
    }
    for std in result.fstdnsim {
        assert_eq!(std, 0.0);
    }
    for each_fvdegenergy in result.fvdegenergy {
        assert_gt!(each_fvdegenergy, 0.0);
    }
}

#[test]
fn test_non48k_sample_rate() {
    let path =
        "test_data/conformance_testdata_subset/non_48k_sample_rate/guitar48_stereo_44100Hz.wav";
    fullband_manager().run(path, path).unwrap();
}

#[test]
fn test_unscaled_speech_mode() {
    let (ref_path, _) = speech_files();
    let result = wideband_manager(true).run(ref_path, ref_path).unwrap();
    // C++ reference: 4.01586 (visqol_cli --use_speech_mode --use_unscaled_speech_mos_mapping
    //                          --use_lattice_model=false)
    assert_abs_diff_eq!(result.moslqo, 4.01586, epsilon = 0.001);
}

#[test]
fn test_scaled_speech_mode() {
    let (ref_path, _) = speech_files();
    let result = wideband_manager(false).run(ref_path, ref_path).unwrap();
    assert_abs_diff_eq!(result.moslqo, 5.0, epsilon = 0.001);
}

fn speech_files() -> (&'static str, &'static str) {
    (
        "test_data/clean_speech/CA01_01.wav",
        "test_data/clean_speech/transcoded_CA01_01.wav",
    )
}

fn guitar_files() -> (&'static str, &'static str) {
    (
        "test_data/conformance_testdata_subset/guitar48_stereo.wav",
        "test_data/conformance_testdata_subset/guitar48_stereo_64kbps_aac.wav",
    )
}
