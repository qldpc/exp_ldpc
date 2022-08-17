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
  version = "0.8.2";
  format = "pyproject";
  src = ../../.;
  nativeBuildInputs = with rustPlatform; [ cargoSetupHook maturinBuildHook ];

  cargoDeps = rustPlatform.fetchCargoTarball {
    inherit src;
    name = "${pname}-${version}";
    hash = "sha256-3BTaRFfu06waGx1vYvA2yiauYciQLcsy/GvAKux6Mqk=";
  };

  buildInputs = lib.optionals stdenv.isDarwin [ libiconv ];

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
