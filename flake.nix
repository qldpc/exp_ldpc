{
  description = "A very basic flake";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/release-22.11";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonPackageOverlay = import ./packages; 
          python3 = pkgs.python3.override {
            packageOverrides = pythonPackageOverlay;
          };
        in rec {
          # Basic python interpreter with qldpc package
          packages.default = python3.withPackages (ps: [
            ps.qldpc
            ps.stim
            ps.pytest
          ]);
          # Add useful extra packages
          packages.experiment = python3.withPackages (ps: [
            ps.qldpc
            ps.stim
            ps.ldpc

            ps.pytest
            ps.pdoc3
            
            ps.numpy
            ps.pandas
            ps.scipy
            ps.matplotlib
          ]);
          overlays.python3 = (final: prev: {
            python3 = prev.python3.override {
              packageOverrides = pythonPackageOverlay;
            };
          });
        }
      );
}
