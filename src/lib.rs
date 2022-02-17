use error_correcting_code::{ErrorCorrectingCode,DecoderWrapper};
use first_min_belief_prop::FirstMinBeliefProp;
use first_min_bp_plus_ssf::FirstMinBPplusSSF;
use small_set_flip::SmallSetFlip;
use pyo3::prelude::*;

mod small_set_flip;
mod first_min_belief_prop;
mod first_min_bp_plus_ssf;
mod error_correcting_code;

#[pymodule]
fn qldpc(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ErrorCorrectingCode>()?;
    m.add_class::<DecoderWrapper>()?;
    m.add_class::<FirstMinBeliefProp>()?;
    m.add_class::<SmallSetFlip>()?;
    m.add_class::<FirstMinBPplusSSF>()?;
    Ok(())
}


// Convert Vec<u8> to Vec<bool>
// Take repeated measurement rounds and correct with BP (Mod for single meas tanner graph -> repeated measurement tanner graph?)
// Correct final mesurement round with first min BP + SSF separately
// Apply correction to final readout
// Compute logicals
// Check diff