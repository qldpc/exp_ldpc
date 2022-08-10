{ buildPythonPackage
, fetchFromGitHub
, pytestCheckHook
, numpy
, numba
, typing-extensions
, pytest
}:

buildPythonPackage rec {
  pname = "galois";
  version = "0.0.32";
  format = "pyproject";

  src = fetchFromGitHub {
    owner = "mhostetter";
    repo = "galois";
    rev = "bf1275a815fd21162198ce788633294ad97b4675";
    sha256 = "sha256-+cxRLrfqk3N9pWKCVsTxruZwMYZ5dQyKJRnrb8y+ECM=";
  };

  propagatedBuildInputs = [
    numpy
    numba
    typing-extensions
    pytest
  ];

  checkInputs = [
    pytestCheckHook
  ];

  checkImport = [ "galois" ];

  enableParallelBuilding = true;

  postPatch = ''
    substituteInPlace setup.cfg \
    --replace "numpy >= 1.18.4, < 1.23" "numpy >= 1.18.4, < 1.24"
  '';

  doCheck = false;
}
