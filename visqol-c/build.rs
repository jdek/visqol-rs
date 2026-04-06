use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let header = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_language(cbindgen::Language::C)
        .with_include_guard("VISQOL_H")
        .generate()
        .expect("Unable to generate C bindings");

    // Write into the source tree for local development.
    let src_include = PathBuf::from(&crate_dir).join("include").join("visqol.h");
    header.write_to_file(&src_include);

    // Write into OUT_DIR so nix / packagers can pick it up.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    header.write_to_file(out_dir.join("visqol.h"));
}
