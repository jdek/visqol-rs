{
  inputs = {
    flakelight-rust.url = "github:accelbread/flakelight-rust";
    flakelight-rust.inputs.naersk.inputs.nixpkgs.follows = "flakelight-rust/flakelight/nixpkgs";
  };
  outputs = { flakelight-rust, ... }: flakelight-rust ./. ({ lib, src, config, ... }: {
    systems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];

    pname = "visqol";

    fileset = lib.fileset.unions [
      (lib.fileset.fileFilter (file: file.hasExt "rs" || file.name == "Cargo.toml") src)
      (src + /Cargo.lock)
      (lib.fileset.maybeMissing (src + /.cargo/config.toml))
      (src + /model)
    ];

    packages = {
      visqol-c = { naersk, defaultMeta, makePkgconfigItem, symlinkJoin, ... }:
        let
          clib = naersk.buildPackage {
            name = "visqol-c-0.3.1";
            src = lib.fileset.toSource { root = src; inherit (config) fileset; };
            cargoBuildOptions = default: default ++ [ "--package" "visqol-c" ];
            copyLibs = true;
            copyBins = false;
            strictDeps = true;
            postInstall = ''
              mkdir -p $out/include
              find target -name visqol.h -path '*/out/visqol.h' -exec cp {} $out/include/ \;
            '';
          };
          pc = makePkgconfigItem {
            name = "visqol";
            inherit (clib) version;
            description = "C API for the ViSQOL perceptual audio quality metric";
            libs = [ "-L${clib}/lib" "-lvisqol_c" ];
            cflags = [ "-I${clib}/include" ];
          };
        in
        symlinkJoin {
          name = "visqol-c-${clib.version}";
          paths = [ clib pc ];
          meta = defaultMeta;
        };
    };
  });
}
