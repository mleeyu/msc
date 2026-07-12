# { pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-26.05") {}, ...}:

let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-26.05") {
    overlays = [
      (final: prev: {
        R = prev.R.overrideAttrs (old: {
          configureFlags =
            (old.configureFlags or [])
            ++ [ "--enable-memory-profiling" ];
        });
      })
    ];
  };

  rextendr_0_5_0 = pkgs.rPackages.buildRPackage rec {
    name = "rextendr-${version}";
    pname = "rextendr";
    version = "0.5.0";

    src = pkgs.fetchurl {
      url = "https://cran.r-project.org/src/contrib/rextendr_${version}.tar.gz";
      hash = "sha256-RfZfx2bisiiG8uLo+ftZs82jx68Dh44VyD3+zDA6FKs=";
    };

    propagatedBuildInputs = with pkgs.rPackages; [
      brio
      cli
      desc
      dplyr
      glue
      jsonlite
      lifecycle
      pkgbuild
      processx
      rlang
      rprojroot
      stringi
      vctrs
      withr
    ];
  };
in pkgs.mkShell {
  buildInputs = with pkgs; [
    cmake
    # cargo
    # rustc

    (rWrapper.override {
      packages = with rPackages; [
        languageserver
        devtools
        usethis
      ] ++ [
        rextendr_0_5_0
      ];
    })
  ];
}
