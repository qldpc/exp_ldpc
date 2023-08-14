{ lib
, stdenv
, buildPythonPackage
, rustPlatform
, pytestCheckHook
, pythonRelaxDepsHook
, numpy
, scipy
, networkx
, galois
, libiconv
}:

buildPythonPackage rec {
  pname = "qldpc";
  version = "0.10.0";
  format = "pyproject";
  src = ../../../.;
  nativeBuildInputs = with rustPlatform; [ cargoSetupHook maturinBuildHook ];

  cargoDeps = rustPlatform.fetchCargoTarball {
    inherit src;
    name = "${pname}-${version}";
    hash = "sha256-1PzQMTklLRtY5Wkh43J5CYQ5z85LA6eFbZ63FMnxkZU=";
  };

  buildInputs = lib.optionals stdenv.isDarwin [ libiconv ];

  pythonRelaxDeps = [ "galois" "numpy" "networkx"];
  
  propagatedBuildInputs = [
    numpy
    scipy
    networkx
    galois
  ];

  checkInputs = [
    pytestCheckHook
  ];
  # Currently blows up due to numba
  doCheck = false;
}
