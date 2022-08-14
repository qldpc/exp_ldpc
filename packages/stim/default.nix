{ lib
, pkgs
, buildPythonPackage
, pythonOlder
, pytestCheckHook
, pytest-xdist
, fetchFromGitHub
, numpy
, pybind11
, cirq
}:

buildPythonPackage rec {
  pname = "stim";
  version = "1.9.0";
  format = "pyproject";

  disabled = pythonOlder "3.6";

  src = pkgs.fetchFromGitHub {
    owner = "quantumlib";
    repo = "Stim";
    rev = "refs/tags/v${version}";
    sha256 = "sha256-zXWdJjFkf74FCWxyVMF8dx0P8GmUkuHFxUo5wYNU2o0=";
  };

  propagatedBuildInputs = [
    numpy
    pybind11
  ];

  checkInputs = [
    pytestCheckHook
    pytest-xdist
  ];

  meta = {
    description = "A tool for high performance simulation and analysis of quantum stabilizer circuits, especially quantum error correction (QEC) circuits.";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ chrispattison ];
    homepage = "https://github.com/quantumlib/stim";
  };

  checkImport = [ "stim" ];

  enableParallelBuilding = true;
  doCheck = true;

  # Requires various extra deps
  disabledTestPaths = [
    # Cirq
    "glue/cirq/stimcirq"
    # Matplotlib
    "glue/sample/src/sinter/plotting_test.py"
    "glue/sample/src/sinter/main_test.py"
    # Networkx
    "glue/zx/stimzx/_external_stabilizer_test.py"
    "glue/zx/stimzx/_text_diagram_parsing_test.py"
    "glue/zx/stimzx/_zx_graph_solver_test.py"
    # Pymatching
    "glue/sample/src/sinter/decoding_test.py"
    "glue/sample/src/sinter/predict_test.py"
    "glue/sample/src/sinter/collection_test.py"
    "glue/sample/src/sinter/collection_work_manager.py"
    # Scipy
    "glue/sample/src/sinter/probability_util_test.py"
    "glue/sample/src/sinter/worker_test.py"
    # Pandas
    "glue/sample/src/sinter/main_combine.py"
  ];
}