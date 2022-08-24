use itertools::Itertools;
use std::ops::{BitXorAssign, BitAnd, BitOr};
use pyo3::prelude::*;
use numpy::ndarray::Array2;
use numpy::{PyArray, PyArray2, PyReadonlyArray2};

type Word = u64;
const WORD_BITS : usize = Word::BITS as usize;

#[derive(Debug, Clone, Copy)]
struct ColSpec {
    word : usize,
    bit  : usize,
}

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
#[target_feature(enable = "avx")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "bmi2")]
unsafe fn _row_reduce(a : &mut Vec::<Word>, lda : usize, col_words : usize, last_word_cols : usize) {
    let num_cols = (col_words-1)*WORD_BITS + last_word_cols;
    // Row to put the pivot on
    let mut r = 0usize;
    for k in 0usize..col_words {
        for k_bit in if k+1 < col_words { 0usize..WORD_BITS } else { 0usize..last_word_cols } {
            let k_col_spec = ColSpec{word:k, bit:k_bit};
            let pivot = find_pivot(a, lda, r, k_col_spec);
            if let Some(pivot_src) = pivot {
                reduce_col(a, lda, col_words, pivot_src, r, k_col_spec);
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
#[inline(always)]
unsafe fn find_pivot(a : &Vec<Word>, lda : usize, start_row: usize, col : ColSpec) -> Option<usize> {
    // TODO: Rewrite this to not be ugly
    let col_mask = (1 as Word).unchecked_shl(col.bit as Word);
    for i in start_row..lda {
        if a.get_unchecked(i + col.word*lda).bitand(col_mask) != 0 {
            return Some(i);
        }
    }
    None
}

#[inline(always)]
unsafe fn reduce_col(a : &mut Vec::<Word>, lda : usize, cols : usize, pivot_src : usize, pivot_i : usize, pivot_j : ColSpec) {
    // Swap pivot if applicable
    if pivot_src != pivot_i {
        for j in (pivot_j.word..cols).rev() {
            let pivot_row_data = *a.get_unchecked(pivot_src + j*lda);
            *a.get_unchecked_mut(pivot_src + j*lda) = *a.get_unchecked_mut(pivot_i + j*lda); 
            *a.get_unchecked_mut(pivot_i + j*lda) = pivot_row_data;
        }
    }

    let col_mask = (1 as Word).unchecked_shl(pivot_j.bit as Word);
    for i in pivot_i+1 .. lda {
        // Use sparsity to shortcut an O(n) step
        if a.get_unchecked(i + pivot_j.word*lda).bitand(col_mask) != 0 {
            for j in pivot_j.word..cols {
                let pivot_row_data = *a.get_unchecked(pivot_src + j*lda);
                a.get_unchecked_mut(i + j*lda).bitxor_assign(pivot_row_data);
            }
        } 
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use test::Bencher;

    use rand::SeedableRng;
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;

    extern crate test;


    #[bench]
    fn bench_gf2_row_reduce(bench : &mut Bencher) {
        let sparsity = 0.005;
        let n = 8192;
        let rows = n;
        let cols = (n+WORD_BITS-1)/WORD_BITS;
        let lda = rows;

        let mut rng = ChaCha8Rng::seed_from_u64(0xeface14cd35a75b5);

        // Generate matrix
        let mut data : Vec<Word> = vec![0; rows*cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i + j*lda] = (0usize..WORD_BITS)
                    .map(|k| if rng.gen::<f64>() < sparsity { 1 << k } else { 0 })
                    .fold(0, |a, b| a.bitor(b));
            }
        }

        bench.iter(|| unsafe {
                _row_reduce(&mut data, rows, cols-1, WORD_BITS)
            }
        );
        assert!(data.iter().sum::<Word>() > 0);
    }
}
