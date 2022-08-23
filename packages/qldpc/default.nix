{ lib
, stdenv
, buildPythonPackage
, makeRustPlatform
, fenix
, pytestCheckHook
, numpy
, scipy
, networkx
, galois
, libiconv
}:

let
  rustPlatform = makeRustPlatform {
    inherit (fenix.complete) cargo rustc;
  };
in buildPythonPackage rec {
  pname = "qldpc";
  version = "0.8.2";
  format = "pyproject";
  src = ../../.;
  nativeBuildInputs = with rustPlatform; [ cargoSetupHook maturinBuildHook ];

  cargoDeps = rustPlatform.fetchCargoTarball {
    inherit src;
    name = "${pname}-${version}";
    hash = "sha256-C4ts0rp4FI8fs2royo1zT9Idfc2pZ5UUEn9Jo0Z8qME=";
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
