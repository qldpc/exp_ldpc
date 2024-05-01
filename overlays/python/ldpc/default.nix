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
  version = "0.1.51";

  src = pkgs.fetchFromGitHub {
    owner = "quantumgizmos";
    repo = "ldpc";
    rev = "cb462f0e262a7856796ebaef9a68181f5ffcddb1";
    sha256 = "sha256-j07rwDKtjq9J+IoI7+fxyOm5hLmhGQRsZ8B0HZ5CMCI=";
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
