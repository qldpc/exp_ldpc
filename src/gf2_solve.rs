use itertools::Itertools;
use std::ops::{BitXorAssign, BitAnd};
use pyo3::prelude::*;
use numpy::{PyArray, PyArray1, PyArray2, PyReadonlyArray2, npyffi::NPY_ORDER};

#[pyfunction]
pub fn row_reduce<'py>(py: Python<'py>, py_a : PyObject) -> PyResult<&'py PyArray2<u8>> {
    let a : PyReadonlyArray2<u8> = py_a.extract(py)?;
    let rows = a.shape()[0];
    let cols = a.shape()[1];
    let mut internal = (0..cols).cartesian_product(0..rows).map(|(j,i)| 
        *a.readonly().get([i,j]).unwrap()).collect::<Vec<u8>>();
    _row_reduce(&mut internal, rows, cols);
    // for j in 0..cols {
    //     for i in 0..rows {
    //         // Why do we need unsafe here?
    //         unsafe { *a.uget_mut([i,j]) = internal[i + j*rows]; }
    //     }
    // }
    let result = PyArray1::<u8>::from_vec(py, internal)
        .reshape_with_order([rows,  cols], NPY_ORDER::NPY_FORTRANORDER)?;
    Ok(result)
}
    
// Row reduce this in place
// We use Fortran ordering of words (column-major)
// But each word is bit-packed entries of a row
// No bitpacking for now
fn _row_reduce(a : &mut Vec::<u8>, lda : usize, cols : usize) {
    // Row to put the pivot on
    let mut r = 0usize;
    for k in 0usize..cols {
        let pivot = find_pivot(a, lda, k, r);
        if let Some(pivot_src) = pivot {
            assert!(a[pivot_src + k*lda] == 1);
            reduce_col(a, lda, cols, pivot_src, (r, k));
            r += 1;
        }

        if r >= cols {
            break;
        }
    }
}

/// Find a pivot row in col in the range [start_row, lda)
fn find_pivot(a : &Vec<u8>, lda : usize, col : usize, start_row: usize) -> Option<usize> {
    // TODO: Rewrite this to not be ugly
    for i in start_row..lda {
        if a[i + col*lda] != 0 {
            return Some(i);
        }
    }
    None
}

fn reduce_col(a : &mut Vec::<u8>, lda : usize, cols : usize, pivot_src : usize, (pivot_i, pivot_j) : (usize, usize)) {
    for j in (pivot_j..cols).rev() {
        reduce_col_inner(a, lda, j, pivot_src, (pivot_i, pivot_j));
    }
}

/// Inner loop of row reduction routine
/// Swap the row pivot_src with pivot_row and then use pivot_row to reduce all the other entries in this column (may be bitpacked)
fn reduce_col_inner(a : &mut Vec::<u8>, lda : usize, col : usize, pivot_src : usize, (pivot_i, pivot_j) : (usize, usize)) {
    let pivot_row_data = a[pivot_src + col*lda];
    // Do we need this check?
    if pivot_src != pivot_i {
        a.swap(pivot_src + col*lda, pivot_i + col*lda);
    }

    for i in (0 .. pivot_i).chain(pivot_i+1 .. lda) {
        // Add row pivot_row to row i scaling by the coefficient that makes the bit in the pivot column 0
        // We will need to unpack bits to vectorize
        let reduce_coeff = a[i + pivot_j*lda].bitand(pivot_row_data);
        a[i + col*lda].bitxor_assign(reduce_coeff);
    }
}
