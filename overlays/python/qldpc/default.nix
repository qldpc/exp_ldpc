{ lib
, stdenv
, buildPythonPackage
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
