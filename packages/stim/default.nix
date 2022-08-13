{ lib
, pkgs
, buildPythonPackage
, pytestCheckHook
, pytest-xdist
, fetchFromGitHub
, numpy
, pybind11
, cirq
}:

buildPythonPackage rec {
  pname = "stim";
  version = "1.9.0";
  format = "setuptools";

  src = pkgs.fetchFromGitHub {
    owner = "quantumlib";
    repo = "Stim";
    rev = "refs/tags/v${version}";
    sha256 = "sha256-zXWdJjFkf74FCWxyVMF8dx0P8GmUkuHFxUo5wYNU2o0=";
  };

  propagatedBuildInputs = [
    numpy
    pybind11
  ];

  checkInputs = [
    pytestCheckHook
    pytest-xdist
  ];

  meta = {
    description = "A tool for high performance simulation and analysis of quantum stabilizer circuits, especially quantum error correction (QEC) circuits.";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ chrispattison ];
    homepage = "https://github.com/quantumlib/stim";
  };

  checkImport = [ "stim" ];

  enableParallelBuilding = true;

  doCheck = false;
}