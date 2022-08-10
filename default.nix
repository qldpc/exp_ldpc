{ pkgs ? import <nixpkgs> {} }:
let
  pythonPackages = import ./packages;
  python = pkgs.python39.override {
    packageOverrides = pythonPackages;
  };
in 
  pkgs.mkShell {
    buildInputs = [ (python.withPackages (ps: [
      ps.galois
      ps.stim
      ps.qldpc
    ])) ];
  }
