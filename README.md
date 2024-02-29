# QEC utilities for practical realizations of general qLDPC codes

[![pytest](https://github.com/qldpc/exp_ldpc/actions/workflows/pytest.yml/badge.svg)](https://github.com/qldpc/exp_ldpc/actions/workflows/pytest.yml)

This repository contains (WIP) research code related to practical implementation of general qLDPC codes.

Please look at the scripts in `/scripts` for usage instructions.

### Caveats
There are a few corners of the codebase that are somewhat incomplete, like the Rust decoders, and the documentation is still somewhat sparse

## Documentation
Generate documentation with `pdoc`
```bash
pdoc -o docs --html qldpc
```

## Layout
This is organized as a Python library for generating quantum codes, generating syndrome extraction circuits, and setting up QEC experiments.
In particular, there is a python package `qldpc` and some python scripts to do stuff with it in `/scripts`.

The python package will work without the rust library, but the only installation method is currently through PEP 517 + maturin instead of setuptools.
There is an (incomplete) Rust library for decoding.

The circuits are output in the Cirq language and can be sampled using Stim (see `/scripts`).

## Features
### Available Codes Generators
- General Hypergraph / Homological Product Codes
- Expander Codes (Hypergraph product codes based on random bipartite graphs)
- Lifted product codes based on Tanner codes on Cayley graphs of PSL(2,q)
- Quasicyclic lifted product codes

### Available Decoders (Currently WIP)
- Small Set Flip
- Belief Propagation

### Circuit Generation Capabilities
- Syndrome extraction circuit assuming all to all gates based on edge coloring the Tanner graph

## Installation
This project uses the python package Maturin for installation.

1) Install Rust ex. following https://rustup.rs/
2) Create a python virtual environment
```bash
python -m venv my_python_virtual_environment
```
3) Activate the virtual environment
```bash
source my_python_virtual_environment/bin/activate
```

4) Install the qldpc package into the virtual environment
```bash
git clone https://github.com/qldpc/exp_ldpc.git
cd exp_ldpc
pip install .
```
You can also build with maturin directly ex. `maturin develop --release`


Note: the virtual environment will need to be activated any time you want to use the package (step 3)

## Alternate Installation (Singularity)
The repository contains a container definition that you can use to build a container with the qldpc package installed
```bash
git clone https://github.com/qldpc/exp_ldpc.git
sudo singularity build ldpc.sif exp_ldpc/ldpc.def
```

## Alternate Installation (Nix)
A nix flake is provided:
Use `nix shell .` to spawn a shell where the package is installed in the python interpreter or `nix run . --` to start the interpreter directly.
Use `nix run .#experiment` to spawn an interpreter with a few more useful packages.
Alternatively, build your own python environment by supplying further python package overlays to the default package (see `flake.nix`).

## Generating a code and syndrome extraction circuit
Try `python /scripts/generate_code_circuit.py --help` and `python /scripts/generate_code_circuit.py 4 3 12 --rounds 1 --save_code code.txt --save_circuit circuit.txt`
to generate a (3,4) hypergraph product code on 225 qubits with 108 X and Z checks and 9 logicals.
Constructing the logicals is a bit tricky and is currently done through Sage.

### Code Format
The routines `qldpc.write_check_generators` and `qldpc.read_check_generators` read and write the code format specification giving out a scipy sparse matrix that is the X and Z check matrices.
The check format is roughly inspired by DIMACS with a header denoting number of qubits, number of X checks, number of Z checks, and then listing the supports of each X and Z check
