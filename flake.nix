{
  description = "Utilies for simulating the performance of decoders for  general quantum LDPC codes";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/release-23.05";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          overlays = import ./overlays;
          pkgs = nixpkgs.legacyPackages.${system}.extend overlays.python;
        in {
          # Basic python interpreter with qldpc package
          packages.default = pkgs.python3.withPackages (ps: [
            ps.qldpc
            ps.stim
            ps.pytest
          ]);
          # Add useful extra packages
          packages.experiment = pkgs.python3.withPackages (ps: [
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
          overlays = overlays;
        }
      );
}
