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
, cvxpy
, libiconv
}:

buildPythonPackage rec {
  pname = "qldpc";
  version = "0.9.0";
  format = "pyproject";
  src = ../../../.;
  nativeBuildInputs = with rustPlatform; [ cargoSetupHook maturinBuildHook ];

  cargoDeps = rustPlatform.fetchCargoTarball {
    inherit src;
    name = "${pname}-${version}";
    hash = "sha256-PEaz36nF4BIEySpgjtiEh965DiaoS+klYtAIjvjOcac=";
  };

  buildInputs = lib.optionals stdenv.isDarwin [ libiconv ];

  pythonRelaxDeps = [ "galois" "numpy" "networkx" "cvxpy"];
  
  propagatedBuildInputs = [
    numpy
    scipy
    networkx
    galois
    cvxpy
  ];

  checkInputs = [
    pytestCheckHook
  ];
  # Currently blows up due to numba
  doCheck = false;
}
