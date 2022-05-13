# QEC utilities for practical realizations of general qLDPC codes

This repository contains (WIP) research code related to practical implementation of general qLDPC codes.
It is distributed under an MIT license and will eventually be publically open source, but for the moment I ask that you ask for permission before sharing the sources.

### Caveats
I have made attempts to test as many pieces as possible, but this is an incomplete research code in an area where many algorithms do not have precise proofs of correctness and thus are difficult to test.

In particular, while much of the python codebase has good test coverage, the decoders are still fairly experimental.
Please be careful with the results as the only *real* test that can be done is a full end to end simulation which this codebase has not yet gone through.

This is likely to change over the coming months as time permits.

## Layout
This is organized as a Python library for generating quantum codes, generating syndrome extraction circuits, and setting up QEC experiments.
In particular, there is a python package `qldpc` and some python scripts to do stuff with it in `/scripts`.
The python package will work without the rust library, but the only installation method is currently through maturin instead of setuptools.

There is an (incomplete) Rust library for decoding.

The circuits are output in the Cirq language and can be sampled using Stim (see `/scripts`).

## Features
### Available Codes Generators
- General Hypergraph / Homological Product Codes
- Expander Codes (Hypergraph product codes based on random bipartite graphs)

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
4) Install maturin into it
```bash
pip install maturin
```
5) Install the qldpc package into the virtual environment
```bash
git clone https://github.com/ChrisPattison/exp_ldpc.git
cd exp_ldpc
```
and
```bash
maturin develop --release
```
or (take your pick)
```bash
maturin build --release
pip install target/wheels/exp_ldpc_0.1.0-*.whl
```

Note: the virtual environment will need to be activated any time you want to use the package (step 3)