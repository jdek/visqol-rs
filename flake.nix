{
  inputs = {
    flakelight-rust.url = "github:accelbread/flakelight-rust";
    flakelight-rust.inputs.naersk.inputs.nixpkgs.follows = "flakelight-rust/flakelight/nixpkgs";
    naersk.follows = "flakelight-rust/naersk";
  };
  outputs =
    { flakelight-rust, naersk, ... }:
    flakelight-rust ./. (
      {
        lib,
        src,
        config,
        ...
      }:
      let
        cargoSrc = lib.fileset.toSource { root = src; inherit (config) fileset; };
        buildVisqolC =
          {
            pkgs,
            tuneNative ? false,
          }:
          let
            inherit (pkgs)
              makePkgconfigItem
              symlinkJoin
              stdenv
              ;
            naersk-lib = pkgs.naersk or (pkgs.callPackage naersk { });
            clib = naersk-lib.buildPackage {
              src = cargoSrc;
              RUSTC_BOOTSTRAP = 1;
              cargoBuildOptions = default: default ++ [ "--package" "visqol-c" ];
              copyLibs = true;
              copyBins = false;
              strictDeps = true;
              CARGO_BUILD_RUSTFLAGS = lib.optionalString tuneNative "-C target-cpu=native";
              postInstall = ''
                mkdir -p $out/include
                find target -name visqol.h -path '*/out/visqol.h' -exec cp {} $out/include/ \;
              ''
              + lib.optionalString stdenv.hostPlatform.isDarwin ''
                install_name_tool -id $out/lib/libvisqol_c.dylib $out/lib/libvisqol_c.dylib
              '';
            };
            pc = makePkgconfigItem {
              name = "visqol";
              version = clib.version;
              description = "C API for the ViSQOL perceptual audio quality metric";
              libs = [
                "-L${clib}/lib"
                "-lvisqol_c"
              ];
              cflags = [ "-I${clib}/include" ];
            };
          in
          symlinkJoin {
            name = "visqol-c-${clib.version}";
            paths = [
              clib
              pc
            ];
          };
      in
      {
        systems = [
          "aarch64-darwin"
          "x86_64-darwin"
          "x86_64-linux"
          "aarch64-linux"
        ];

        rust.enable_unstable = true;

        fileset = lib.fileset.unions [
          (lib.fileset.fileFilter (file: file.hasExt "rs" || file.name == "Cargo.toml") src)
          (src + /Cargo.lock)
          (lib.fileset.maybeMissing (src + /.cargo/config.toml))
          (src + /model)
        ];

        packages = {
          visqol-c = pkgs: buildVisqolC { inherit pkgs; tuneNative = true; };
          visqol-c-aarch64-linux-gnu =
            pkgs: buildVisqolC { pkgs = pkgs.pkgsCross.aarch64-multiplatform; };
          visqol-c-aarch64-linux-musl =
            pkgs: buildVisqolC { pkgs = pkgs.pkgsCross.aarch64-multiplatform-musl; };
        };

        overlays.default = final: _prev: {
          visqol-c = buildVisqolC { pkgs = final; tuneNative = true; };
        };
      }
    );
}
