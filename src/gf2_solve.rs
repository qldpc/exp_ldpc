use itertools::Itertools;
use std::ops::{BitXorAssign, BitAnd, BitOr};
use pyo3::prelude::*;
use numpy::ndarray::Array2;
use numpy::{PyArray, PyArray2, PyReadonlyArray2};

type Word = u128;
const WORD_BITS : usize = Word::BITS as usize;

#[pyfunction]
pub fn row_reduce<'py>(py: Python<'py>, py_a : PyObject) -> PyResult<&'py PyArray2<u8>> {
    let a : PyReadonlyArray2<u8> = py_a.extract(py)?;
    let rows = a.shape()[0];
    let cols = a.shape()[1];
    let lda = rows;
    
    // Round down because of last word cols
    let col_words = (cols-1) / WORD_BITS+ 1;
    let last_word_cols = (cols-1) % WORD_BITS + 1;

    // Pack input
    let mut internal = vec![0; col_words*rows];
    for i in 0..rows {
        for j in 0..col_words {
            // Bitpack matrix
            internal[i + j*lda] = if j+1 < col_words { 0usize..WORD_BITS } else { 0usize..last_word_cols }
                .map(|k| if *a.readonly().get([i,j * WORD_BITS + k]).unwrap() > 0 { 1 << k} else { 0 })
                .fold(0, |a, b| a.bitor(b));
        }
    }

    // Compute
    unsafe {
        _row_reduce(&mut internal, rows, col_words, last_word_cols);
    }
    

    // Unpack result
    let mut result = Array2::zeros((rows,  cols));
    for i in 0..rows {
        for j in 0..col_words {
            let packed_word = internal[i + j*lda];
            for k in if j+1 < col_words { 0usize..WORD_BITS } else { 0usize..last_word_cols } {
                result[[i,j*WORD_BITS + k]] = (packed_word.bitand(1 << k) >> k) as u8;
            }
        }
    }
    Ok(PyArray::from_owned_array(py, result))
}
    
// Row reduce this in place
// We use Fortran ordering of words (column-major)
// But each word is bit-packed entries of a row
// No bitpacking for now
#[inline(never)]
unsafe fn _row_reduce(a : &mut Vec::<Word>, lda : usize, col_words : usize, last_word_cols : usize) {
    let num_cols = (col_words-1)*WORD_BITS + last_word_cols;
    // Row to put the pivot on
    let mut r = 0usize;
    for k in 0usize..col_words {
        for k_bit in if k+1 < col_words { 0usize..WORD_BITS } else { 0usize..last_word_cols } {
            let pivot = find_pivot(a, lda, k, k_bit, r);
            if let Some(pivot_src) = pivot {
                reduce_col(a, lda, col_words, pivot_src, (r, k), k_bit);
                r += 1;
            }

            if r >= num_cols {
                break;
            }
        }
    }
}

// TODO: Is this index arithemtic checked?

/// Find a pivot row in col in the range [start_row, lda)
#[inline]
unsafe fn find_pivot(a : &Vec<Word>, lda : usize, col : usize, col_bit : usize, start_row: usize) -> Option<usize> {
    // TODO: Rewrite this to not be ugly
    let col_mask = (1 as Word).unchecked_shl(col_bit as Word);
    for i in start_row..lda {
        if a.get_unchecked(i + col*lda).bitand(col_mask) != 0 {
            return Some(i);
        }
    }
    None
}

#[inline]
unsafe fn reduce_col(a : &mut Vec::<Word>, lda : usize, cols : usize, pivot_src : usize, (pivot_i, pivot_j) : (usize, usize), pivot_j_bit : usize) {
    for j in (pivot_j..cols).rev() {
        reduce_col_inner(a, lda, j, pivot_src, (pivot_i, pivot_j), pivot_j_bit);
    }
}

/// Inner loop of row reduction routine
/// Swap the row pivot_src with pivot_row and then use pivot_row to reduce all the other entries in this column (may be bitpacked)
#[inline]
unsafe fn reduce_col_inner(a : &mut Vec::<Word>, lda : usize, col : usize, pivot_src : usize, (pivot_i, pivot_j) : (usize, usize), pivot_j_bit : usize) {
    let pivot_row_data = *a.get_unchecked(pivot_src + col*lda);
    // Do we need this check?
    if pivot_src != pivot_i {
        *a.get_unchecked_mut(pivot_src + col*lda) = *a.get_unchecked_mut(pivot_i + col*lda); 
        *a.get_unchecked_mut(pivot_i + col*lda) = pivot_row_data;
    }

    let col_mask = (1 as Word).unchecked_shl(pivot_j_bit as Word);

    // Manually duplicating this ends up being slightly faster (inhibited optimizations?)
    for i in 0 .. pivot_i {
        // If the bit in the pivot column is set, add the pivot row to this one
        if a.get_unchecked(i + pivot_j*lda).bitand(col_mask) != 0 {
            a.get_unchecked_mut(i + col*lda).bitxor_assign(pivot_row_data);
        }        
    }

    for i in pivot_i+1 .. lda {
        if a.get_unchecked(i + pivot_j*lda).bitand(col_mask) != 0 {
            a.get_unchecked_mut(i + col*lda).bitxor_assign(pivot_row_data);
        }   
    }
}
