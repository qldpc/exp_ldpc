{ lib
, pkgs
, buildPythonPackage
, pythonOlder
, pytestCheckHook
, pytest-xdist
, fetchFromGitHub
, numpy
, pybind11
, cirq-core
, matplotlib
, networkx
, scipy
, pandas
}:

buildPythonPackage rec {
  pname = "stim";
  version = "1.9.0";
  format = "pyproject";

  disabled = pythonOlder "3.6";

  src = pkgs.fetchFromGitHub {
    owner = "quantumlib";
    repo = "Stim";
    rev = "2a7faf788195653f7ab55a09fa52a613b8bd1b03";
    sha256 = "sha256-/YVIwxolQTp2G1vB8CAbTsRUrsnznIB1RVTKC1oZvH8=";
  };

  propagatedBuildInputs = [
    numpy
    pybind11
  ];

  checkInputs = [
    pytestCheckHook
    pytest-xdist

    cirq-core
    matplotlib
    networkx
    scipy
    pandas
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
    "glue/sample/src/sinter/_main_test.py"
    "glue/sample/src/sinter/_decoding_test.py"
    "glue/sample/src/sinter/_predict_test.py"
    "glue/sample/src/sinter/_collection_test.py"
    "glue/sample/src/sinter/_collection_work_manager.py"
    "glue/sample/src/sinter/_worker_test.py"
  ];
}
