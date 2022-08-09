{ pkgs ? (import <nixpkgs> {}) }:
let 
    galois = (pkgs.callPackage ./packages/galois.nix {
        pythonPackages = pkgs.python39Packages;
    });

    stim = (pkgs.callPackage ./packages/stim.nix {
        pythonPackages = pkgs.python39Packages;
    });

    repo_packages = with pkgs.python39Packages; [
        pandas
        numpy
        scipy
        matplotlib
        pytest
        networkx
    ];
    
    customPython = pkgs.python39.buildEnv.override {
        extraLibs = repo_packages ++ [ stim galois ];
    };
in
pkgs.mkShell {
    buildInputs = [ customPython ];
}
