use pyo3::prelude::*;
use pyo3::types::PyIterator;

use petgraph::graph::UnGraph;
use enum_as_inner::EnumAsInner;

#[derive(Debug, EnumAsInner)]
pub enum TannerGraphNode {
    CheckNode(usize),
    BitNode(usize),
}

// #[pyclass]
// pub struct ErrorCorrectingCode {
//     tanner_graph : UnGraph<TannerGraphNode, ()>,
// }


// #[pymethods]
// impl ErrorCorrectingCode {
//     #[new]
//     pub fn new(checks : PyIterator) -> Self {
//         let checks = checks.map(
//             |x| PyAny::extract::<PyIterator>(x).map(
//                 |inner| inner.and_then(PyAny::extract::<usize>)
//             ).collect::<Vec<_>>()
//         ).collect::<Vec<_>>();

//         let mut tanner_graph = UnGraph::new();

//         ErrorCorrectingCode {tanner_graph}
//     }
// }