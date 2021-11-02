use pyo3::prelude::*;

mod small_set_flip;
mod error_correcting_code;

#[pyfunction]
fn test_func(val : &str) -> PyResult<usize> {
    Ok(val.len())
}

#[pymodule]
fn expldpc(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_func, m)?)?;

    Ok(())
}
