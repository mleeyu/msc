{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "https://github.com/NixOS/nixpkgs/tarball/nixos-26.05";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
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

    robustMCGARCH = pkgs.rPackages.buildRPackage {
      name = "robustMCGARCH";
      version = "0.0.0.9000";
      src = ./robustMCGARCH;
      postPatch = ''
        patchShebangs .
      '';
      nativeBuildInputs = with pkgs; [
        R

        cargo
        rustc
        pkg-config

        cmake
        nlopt
      ];
      propagatedBuildInputs = with pkgs.rPackages; [
        rextendr_0_5_0
      ];
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        cargo
        rustc
        rust-analyzer

        cmake
        nlopt

        (rWrapper.override {
          packages = with rPackages; [
            languageserver
            devtools
            usethis
          ] ++ [
            arrow
            xts
            rextendr_0_5_0
          ];
        })
      ];

      env.RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
    };

    # To use result symlink, update .libPaths in R:
    # .libPaths(c(.libPaths(), normalizePath("result/library")))
    packages.${system}.default = robustMCGARCH;
  };
}
