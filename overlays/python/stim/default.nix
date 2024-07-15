{ lib
, pkgs
, buildPythonPackage
, pythonOlder
, pytestCheckHook
, pytest-xdist
, fetchFromGitHub
, substituteAll
, numpy
, pybind11
, cirq-core
, matplotlib
, networkx
, scipy
, pandas
, setuptools
}:

buildPythonPackage rec {
  pname = "stim";
  version = "1.13.0";
  format = "pyproject";

  disabled = pythonOlder "3.6";

  src = pkgs.fetchFromGitHub {
    owner = "quantumlib";
    repo = "Stim";
    rev = "refs/tags/v${version}";
    hash = "sha256-anJvDHLZ470iNw0U7hq9xGBacDgqYO9ZcmmdCt9pefg=";
  };

  patches = [
    (substituteAll {
      src = ./0001-relax-pybind11-dep-in-pyproject.toml.patch;
    })
  ];

  nativeBuildInputs = [
    setuptools
  ];
  
  propagatedBuildInputs = [
    numpy
    pybind11
  ];

  nativeCheckInputs = [
    pytestCheckHook
    pytest-xdist

    cirq-core
    matplotlib
    networkx
    scipy
    pandas
  ];

  pythonRelaxDeps = [
    "pybind11"
  ];

  meta = {
    description = "A tool for high performance simulation and analysis of quantum stabilizer circuits, especially quantum error correction (QEC) circuits.";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ chrispattison ];
    homepage = "https://github.com/quantumlib/stim";
  };

  pythonImportsCheck = [ "stim" ];

  enableParallelBuilding = true;

  disabledTestPaths = [
    # No pymatching
    "glue/sample/src/sinter/"
    # Broken due to some networkx incompatibility?
    "glue/zx/stimzx/_text_diagram_parsing_test.py"
  ];
}
