use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = PathBuf::from(&crate_dir);

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_language(cbindgen::Language::C)
        .with_include_guard("VISQOL_H")
        .generate()
        .expect("Unable to generate C bindings")
        .write_to_file(out_dir.join("include").join("visqol.h"));
}
