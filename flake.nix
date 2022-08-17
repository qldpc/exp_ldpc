{
  description = "A very basic flake";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonPackageOverlay = import ./packages; 
          python = pkgs.python39.override {
            packageOverrides = pythonPackageOverlay;
          };
        in rec {
          # Use nix run . to start a python interpreter with the package
          packages.default = python.withPackages (ps: [
            ps.qldpc
            ps.stim
            ps.pytest
          ]);

          # Or add it to a python environment with other packages
          packages.experiment = python.withPackages (ps: [
            ps.qldpc
            ps.stim
            ps.ldpc

            ps.numpy
            ps.pandas
            ps.scipy
            ps.matplotlib
          ]);
        }
      );
}
