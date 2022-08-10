{ lib
, stdenv
, buildPythonPackage
, rustPlatform
, pytestCheckHook
, numpy
, scipy
, networkx
, galois
, libiconv
}:

buildPythonPackage rec {
  pname = "qldpc";
  version = "0.6.1";
  format = "pyproject";
  src = ../../.;
  nativeBuildInputs = with rustPlatform; [ cargoSetupHook maturinBuildHook ];

  cargoDeps = rustPlatform.fetchCargoTarball {
    inherit src;
    name = "${pname}-${version}";
    hash = "sha256-iAHrHJte1UeCHYtYLCqXSPeVDjW9tWVMPM+JmeqDIqM=";
  };

  buildInputs = lib.optionals stdenv.isDarwin [ libiconv ];

  propagatedBuildInputs = [
    numpy
    scipy
    networkx
    galois
  ];

  checkInputs = [
    # pytestCheckHook
  ];
}
