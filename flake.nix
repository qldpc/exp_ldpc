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
        in
        {
          # devShell for nix develop
          devShells.default = (pkgs.callPackage ./default.nix { });
          # Overlay for building your own python environment 
          # Note this is an overlay of _python packages_
          # See how it is used/added to in default.nix
          overlays.default = pythonPackageOverlay;
        }
      );
}
