{ pkgs ? import <nixpkgs> {} }:
let
  pythonPackageOverlay = import ./packages; 
  python = pkgs.python39.override {
    packageOverrides = pythonPackageOverlay;
  };
in 
  python.withPackages (ps: [
      ps.galois
      ps.stim
      ps.qldpc
      ps.ldpc
      ps.pytest
    ])