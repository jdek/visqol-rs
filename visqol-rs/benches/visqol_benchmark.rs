use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use visqol_rs::{
    audio_utils,
    constants::{DEFAULT_WINDOW_SIZE, NUM_BANDS_SPEECH},
    variant::Variant,
    visqol_manager::VisqolManager,
};

fn bench_end_to_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e");

    group.bench_function("speech_16k", |b| {
        b.iter(|| {
            let mut visqol = VisqolManager::<NUM_BANDS_SPEECH>::new(
                Variant::Wideband {
                    use_unscaled_mos_mapping: false,
                },
                DEFAULT_WINDOW_SIZE,
            );
            visqol
                .run(
                    "test_data/clean_speech/reference_signal_16k.wav",
                    "test_data/clean_speech/degraded_signal_16k.wav",
                )
                .unwrap()
        })
    });

    group.bench_function("speech_48k", |b| {
        b.iter(|| {
            let mut visqol = VisqolManager::<NUM_BANDS_SPEECH>::new(
                Variant::Wideband {
                    use_unscaled_mos_mapping: false,
                },
                DEFAULT_WINDOW_SIZE,
            );
            visqol
                .run(
                    "test_data/clean_speech/reference_signal.wav",
                    "test_data/clean_speech/degraded_signal.wav",
                )
                .unwrap()
        })
    });

    group.finish();
}

fn bench_load_audio(c: &mut Criterion) {
    c.bench_function("load_mono_16k", |b| {
        b.iter(|| {
            audio_utils::load_as_mono("test_data/clean_speech/reference_signal_16k.wav").unwrap()
        })
    });

    c.bench_function("load_mono_48k", |b| {
        b.iter(|| {
            audio_utils::load_as_mono("test_data/clean_speech/reference_signal.wav").unwrap()
        })
    });
}

fn bench_prepare_and_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_ref");

    group.bench_function("prepare_16k", |b| {
        b.iter(|| {
            let mut visqol = VisqolManager::<NUM_BANDS_SPEECH>::new(
                Variant::Wideband {
                    use_unscaled_mos_mapping: false,
                },
                DEFAULT_WINDOW_SIZE,
            );
            visqol
                .prepare_reference("test_data/clean_speech/reference_signal_16k.wav")
                .unwrap()
        })
    });

    group.bench_function("run_with_ref_16k", |b| {
        let mut visqol = VisqolManager::<NUM_BANDS_SPEECH>::new(
            Variant::Wideband {
                use_unscaled_mos_mapping: false,
            },
            DEFAULT_WINDOW_SIZE,
        );
        let prepared = visqol
            .prepare_reference("test_data/clean_speech/reference_signal_16k.wav")
            .unwrap();
        b.iter(|| {
            visqol
                .run_with_reference(
                    &prepared,
                    "test_data/clean_speech/degraded_signal_16k.wav",
                )
                .unwrap()
        })
    });

    group.finish();
}

fn bench_compute_only(c: &mut Criterion) {
    // Pre-load audio to isolate computation from I/O
    let mut ref_signal =
        audio_utils::load_as_mono("test_data/clean_speech/reference_signal_16k.wav").unwrap();
    let mut deg_signal =
        audio_utils::load_as_mono("test_data/clean_speech/degraded_signal_16k.wav").unwrap();

    c.bench_function("compute_only_16k", |b| {
        b.iter(|| {
            let mut r = ref_signal.clone();
            let mut d = deg_signal.clone();
            let mut visqol = VisqolManager::<NUM_BANDS_SPEECH>::new(
                Variant::Wideband {
                    use_unscaled_mos_mapping: false,
                },
                DEFAULT_WINDOW_SIZE,
            );
            visqol.compute_results(&mut r, &mut d).unwrap()
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_end_to_end, bench_load_audio, bench_prepare_and_reuse, bench_compute_only
}
criterion_main!(benches);
