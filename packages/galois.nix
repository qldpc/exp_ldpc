{ lib
, pkgs
, pythonPackages
, fetchFromGitHub
}:

pythonPackages.buildPythonPackage rec {
  pname = "galois";
  version = "0.0.32";

  disabled = pythonPackages.pythonOlder "3.7";

  src = pkgs.fetchFromGitHub {
    owner = "mhostetter";
    repo = "galois";
    rev = "bf1275a815fd21162198ce788633294ad97b4675";
    sha256 = "sha256-+cxRLrfqk3N9pWKCVsTxruZwMYZ5dQyKJRnrb8y+ECM=";
  };

  propagatedBuildInputs = with pythonPackages; [
    numpy
    numba
    typing-extensions
    pytest
  ];

  checkInputs = [
    pythonPackages.pytestCheckHook
  ];

  meta = {
    description = "The galois library is a Python 3 package that extends NumPy arrays to operate over finite fields.";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ ];
    homepage = "https://github.com/mhostetter/galois";
  };

  doCheck = true;
}
