use crate::error_correcting_code::{Decoder, ErrorCorrectingCode, Bitstring};
use crate::first_min_belief_prop::FirstMinBeliefProp;
use crate::small_set_flip::SmallSetFlip;

use pyo3::exceptions::PyRuntimeError;
use pyo3::{pyclass, pymethods, PyResult};

/// First-min Belief Propagation + Small Set Flipfrom
/// Grospellier et al., Quantum 5, 432 (2021).
#[derive(Debug, Clone)]
#[pyclass]
pub struct FirstMinBPplusSSF {
    belief_prop : FirstMinBeliefProp,
    small_set_flip : SmallSetFlip,
}

impl FirstMinBPplusSSF {
    pub fn new(code : &ErrorCorrectingCode, error_prior : f64) -> Self {
        FirstMinBPplusSSF {
            belief_prop:FirstMinBeliefProp::new(code, error_prior),
            small_set_flip:SmallSetFlip::new(code),
        }
    }
}

#[pymethods]
impl FirstMinBPplusSSF {
    #[new]
    pub fn pynew(code : &ErrorCorrectingCode, error_prior : f64) -> PyResult<Self> {
        Ok(FirstMinBPplusSSF {
            belief_prop:FirstMinBeliefProp::new(code, error_prior),
            small_set_flip:SmallSetFlip::new(code),
        })
    }
}

impl Decoder for FirstMinBPplusSSF {
    fn correct_syndrome(&mut self, syndrome : &mut Bitstring, correction : &mut Bitstring) {
        // Apply BP then SSF
        // The syndrome value is required to be consistent with the input syndrome and output correction so we can compose
        self.belief_prop.correct_syndrome(syndrome, correction);
        self.small_set_flip.correct_syndrome(syndrome, correction);
    }
}