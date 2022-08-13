{ lib
, pkgs
, pythonPackages
, pytestCheckHook
, fetchFromGitHub
, pytest-xdist
, cython
, numpy
, scipy
, tqdm
}:

pythonPackages.buildPythonPackage rec {
  pname = "ldpc";
  version = "1.9.0";

  src = pkgs.fetchFromGitHub {
    owner = "quantumgizmos";
    repo = "ldpc";
    rev = "7909a97d4469dc0caa1eb2c9ac3fc0e0b6abb819";
    sha256 = "sha256-/8ae64xy3IAR/0NLiKRmcZAKh/uLO5oDvxn46lNuWZI=";
  };

  buildInputs = [
    cython
  ];

  propagatedBuildInputs = [
    numpy
    scipy
    tqdm
  ];

  checkInputs = [
    pytestCheckHook
    pytest-xdist
  ];

  meta = {
    description = "This module provides a suite of tools for building and benmarking low density parity check (LDPC) codes";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ chrispattison ];
    homepage = "https://github.com/quantumgizmos/ldpc";
  };

  checkImport = [ "ldpc" ];

  enableParallelBuilding = true;
}
