use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use numpy::PyArray1;

use std::collections::HashSet;

use petgraph::{graph::DiGraph, visit::EdgeRef};
use enum_as_inner::EnumAsInner;

pub type TannerGraph = DiGraph<TannerGraphNode, ()>;

#[derive(Debug, EnumAsInner, Clone)]
pub enum TannerGraphNode {
    CheckNode(usize),
    BitNode(usize),
}

#[pyclass]
pub struct ErrorCorrectingCode {
    pub tanner_graph : TannerGraph,
    pub logicals : Vec<Vec<usize>>,
    pub num_qubits : usize,
}

pub type Bitstring = Vec<bool>;

pub trait Decoder {
    /// A syndrome bit is true if it is non-trivial
    /// A correction bit is true if it supports a non-trivial correction
    /// The syndrome vector will contain the resulting syndrome after the recovery operator in correction is applied
    fn correct_syndrome(self : &mut Self, syndrome : &mut Bitstring, correction : &mut Bitstring);
}

#[pymethods]
impl ErrorCorrectingCode {
    #[new]
    pub fn new(checks : Vec<Vec<usize>>, logicals : Vec<Vec<usize>>) -> PyResult<Self> {

        let num_checks = checks.len();

        // Consistency checks
        if num_checks == 0 { Err(PyErr::new::<PyRuntimeError, _>("Number of checks must be positive")) } else { Ok(()) }?;
        checks.iter().map(|x| if x.len() == 0 { 
            Err(PyErr::new::<PyRuntimeError, _>("Found empty check in list of checks")) 
        } else { Ok(()) }).fold(Ok(()), |a : PyResult<_>, b| { a?; b?; Ok(()) })?;

        let data_indices = checks.iter().map(|x| x.iter()).flatten().map(|x| *x).collect::<HashSet<_>>();
        let num_qubits = data_indices.iter().max().unwrap() + 1;
        
        if logicals.iter().map(|x| x.iter().max().unwrap()).max().map_or(false, |x| *x >= num_qubits) { 
            Err(PyErr::new::<PyRuntimeError, _>("Invalid qubit index in logical")) } else { Ok(()) }?;

        if (0..num_qubits).collect::<HashSet<_>>().difference(&data_indices).count() > 0 { 
            Err(PyErr::new::<PyRuntimeError, _>("Data bit indices not contiguous")) } else { Ok(()) }?;

        // Build tanner graph
        let mut tanner_graph = DiGraph::default();

        let check_node_indices = (0..num_checks).map(|i| tanner_graph.add_node(TannerGraphNode::CheckNode(i))).collect::<Vec<_>>();
        let bit_node_indices = (0..num_qubits).map(|i| tanner_graph.add_node(TannerGraphNode::BitNode(i))).collect::<Vec<_>>();

        for (check_idx, check_support) in checks.iter().enumerate() {
            for bit_idx in check_support.iter() {
                tanner_graph.add_edge(check_node_indices[check_idx], bit_node_indices[*bit_idx], ());
            }
        }

        Ok(ErrorCorrectingCode {tanner_graph, logicals, num_qubits})
    }

    pub fn apply_correction(&self, _py: Python<'_>, correction : &PyArray1<bool>, measurement_outcomes : &PyArray1<bool>) -> PyResult<()> {
        if correction.shape() != measurement_outcomes.shape() {
            Err(PyErr::new::<PyRuntimeError, _>("Correction does not match measurement length"))
        } else {
            for (x, v) in correction.iter().unwrap().zip(measurement_outcomes.iter().unwrap()) {
                *v ^= *x;
            }
            Ok(())
        }
    }

    pub fn measure_logicals<'pyl>(&self, py: Python<'pyl>, measurement_outcomes : &PyArray1<bool>) -> PyResult<&'pyl PyArray1<bool>> {
        if measurement_outcomes.shape()[0] != self.num_qubits { 
            Err(PyErr::new::<PyRuntimeError, _>("Measurement number does not match number of qubits")) } else { Ok(()) }?;
        
        // TODO: This is quadratic time so instrument overhead later
        Ok(PyArray1::from_iter(py, self.logicals.iter().map(|x| x.iter()
            .map(|i| *measurement_outcomes.readonly().get(*i).unwrap()).fold(false, |a,b| a^b))))
    }

}

// ApplyDecoder(ErrorCorrectingCode, Decoder)

/// Returns true if edges are directed from checks to bits
pub fn tanner_graph_edge_orientation(tanner_graph : &TannerGraph) -> bool {
    tanner_graph.edge_references().all(|edge_ref|
           tanner_graph[edge_ref.source()].as_check_node().is_some()
        && tanner_graph[edge_ref.target()].as_bit_node().is_some()
    )
}
