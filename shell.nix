{ pkgs ? (import <nixpkgs> {}) }:
let 
    pythonPackages = pkgs.python39Packages;
    rustPlatform = pkgs.rustPlatform;
    lib = pkgs.lib;
    stdenv = pkgs.stdenv;

    galois = (pkgs.callPackage ./packages/galois.nix { });
    stim = (pkgs.callPackage ./packages/stim.nix { });

    repo_packages = with pythonPackages; [
        pandas
        numpy
        scipy
        matplotlib
        pytest
        networkx
    ];

    qldpc = pythonPackages.buildPythonPackage rec {
        pname = "qldpc";
        version = "0.6.0";
        format = "pyproject";
        src = ./.;
        nativeBuildInputs = with rustPlatform; [ cargoSetupHook maturinBuildHook ];

        cargoDeps = rustPlatform.fetchCargoTarball {
            inherit src;
            name = "${pname}-${version}";
            hash = "sha256-lNBy6uJBPWcttj2ZCGNBfErcInwp9nUbC7UNIK4ZASo=";
        };

        buildInputs = lib.optionals stdenv.isDarwin [ pkgs.libiconv ];

        propagatedBuildInputs = [
            pythonPackages.numpy
            pythonPackages.scipy
            pythonPackages.networkx
            galois
        ];

        checkInputs = [
            pythonPackages.pytestCheckHook
        ];
    };
    
    customPython = pkgs.python39.buildEnv.override {
        extraLibs = repo_packages ++ [ stim galois qldpc ];
    };
in

pkgs.mkShell {
    buildInputs = [ customPython ];
}
