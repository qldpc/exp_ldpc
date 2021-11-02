use pyo3::prelude::*;

#[pyfunction]
fn test_func(val : &str) -> PyResult<usize> {
    Ok(val.len())
}

#[pymodule]
fn expldpc(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_func, m)?)?;

    Ok(())
}
