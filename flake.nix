{
  inputs = {
    flakelight-rust.url = "github:accelbread/flakelight-rust";
    flakelight-rust.inputs.naersk.inputs.nixpkgs.follows = "flakelight-rust/flakelight/nixpkgs";
  };
  outputs = { flakelight-rust, ... }: flakelight-rust ./. ({ lib, src, config, ... }: {
    systems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];

    fileset = lib.fileset.unions [
      (lib.fileset.fileFilter (file: file.hasExt "rs" || file.name == "Cargo.toml") src)
      (src + /Cargo.lock)
      (lib.fileset.maybeMissing (src + /.cargo/config.toml))
      (src + /model)
    ];

    packages = {
      visqol-c = { naersk, defaultMeta, ... }:
        naersk.buildPackage {
          src = lib.fileset.toSource { root = src; inherit (config) fileset; };
          cargoBuildOptions = default: default ++ [ "--package" "visqol-c" ];
          copyLibs = true;
          copyBins = false;
          strictDeps = true;
          meta = defaultMeta;
        };
    };
  });
}
