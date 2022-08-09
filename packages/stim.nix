{ lib
, pkgs
, pythonPackages
, fetchFromGitHub
}:

pythonPackages.buildPythonPackage rec {
  pname = "stim";
  version = "1.9.0";

  src = pkgs.fetchFromGitHub {
    owner = "quantumlib";
    repo = "Stim";
    rev = "a995376f5137001a0590adc0bdf3207fc2446ab0";
    sha256 = "sha256-zXWdJjFkf74FCWxyVMF8dx0P8GmUkuHFxUo5wYNU2o0=";
  };

  propagatedBuildInputs = with pythonPackages; [
    pybind11
    pytest
  ];

  checkInputs = [
    pythonPackages.pytestCheckHook
  ];

  meta = {
    description = "Stim is a tool for high performance simulation and analysis of quantum stabilizer circuits, especially quantum error correction (QEC) circuits.";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ ];
    homepage = "https://github.com/quantumlib/stim";
  };

  doCheck = true;
}
