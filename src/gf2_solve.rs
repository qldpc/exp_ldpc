use itertools::Itertools;
use std::ops::{BitXorAssign, BitAnd, BitOr};
use pyo3::prelude::*;
use numpy::ndarray::Array2;
use numpy::{PyArray, PyArray2, PyReadonlyArray2};
use std::simd::*;

type Word = u64;
const WORD_BITS : usize = Word::BITS as usize;
const SIMD_LANES : usize = 8;
const SIMD_BITS : usize = SIMD_LANES*WORD_BITS;
type SimdWord = Simd<Word, SIMD_LANES>;

#[derive(Debug, Clone, Copy)]
struct ColSpec {
    word : usize,
    lane : usize,
    bit  : usize,
}

fn ceil_div(a : usize, b : usize) -> usize{
    return (a+b-1)/b
}

#[pyfunction]
pub fn row_reduce<'py>(py: Python<'py>, py_a : PyObject) -> PyResult<&'py PyArray2<u8>> {
    let a : PyReadonlyArray2<u8> = py_a.extract(py)?;
    let rows = a.shape()[0];
    let cols = a.shape()[1];
    let lda = rows;
    
    // Round down because of last word cols
    let col_words = (cols-1) / WORD_BITS + 1;
    let last_word_cols = (cols-1) % WORD_BITS + 1;
    let col_simd_words = ceil_div(col_words, SIMD_LANES);

    // Past the end index for the last column
    // let num_cols = ColSpec{};

    // Pack input
    let mut internal = vec![Simd::splat(0); col_simd_words*SIMD_LANES*rows];
    for i in 0..rows {
        for j in 0..col_words {
            // Bitpack matrix
            internal[i + (j/SIMD_LANES)*lda][j%SIMD_LANES] = if j+1 < col_words { 0usize..WORD_BITS } else { 0usize..last_word_cols }
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
unsafe fn _row_reduce(a : &mut Vec::<SimdWord>, lda : usize, col_words : usize, last_word_cols : usize) {
    let num_cols = (col_words-1)*WORD_BITS + last_word_cols;
    // Row to put the pivot on
    let mut r = 0usize;
    for k in 0usize..col_words {
        for k_lane in 0usize .. SIMD_LANES {
            for k_bit in if k+1 < col_words { 0usize..WORD_BITS } else { 0usize..last_word_cols } {
                let k_col_spec = ColSpec{word:k, lane:k_lane, bit:k_bit};
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
}

// TODO: Is this index arithemtic checked?

/// Find a pivot row in col in the range [start_row, lda)
#[inline(always)]
unsafe fn find_pivot(a : &Vec<SimdWord>, lda : usize, start_row: usize, col : ColSpec) -> Option<usize> {
    // TODO: Rewrite this to not be ugly
    let col_mask = (1 as Word).unchecked_shl(col.bit as Word);
    for i in start_row..lda {
        if a.get_unchecked(i + col.word*lda)[col.lane].bitand(col_mask) != 0 {
            return Some(i);
        }
    }
    None
}

#[inline(always)]
unsafe fn reduce_col(a : &mut Vec::<SimdWord>, lda : usize, cols : usize, pivot_src : usize, pivot_i : usize, pivot_j : ColSpec) {
    for j in (pivot_j.word..cols).rev() {
        // Inner loop of row reduction routine
        // Swap the row pivot_src with pivot_row and then use pivot_row to reduce all the other entries in this column (may be bitpacked)

        let pivot_row_data = *a.get_unchecked(pivot_src + j*lda);
        // Do we need this check?
        if pivot_src != pivot_i {
            *a.get_unchecked_mut(pivot_src + j*lda) = *a.get_unchecked_mut(pivot_i + j*lda); 
            *a.get_unchecked_mut(pivot_i + j*lda) = pivot_row_data;
        }

        let col_mask = (1 as Word).unchecked_shl(pivot_j.bit as Word);

        // Manually duplicating this ends up being slightly faster (inhibited optimizations?)        
        // Reduce stuff below the pivot
        for i in pivot_i+1 .. lda {
            if a.get_unchecked(i + pivot_j.word*lda)[pivot_j.lane].bitand(col_mask) != 0 {
                a.get_unchecked_mut(i + j*lda).bitxor_assign(pivot_row_data);
            }   
        }

        // Reduce stuff above the pivot
        // TODO: Defer this until the end
        // for i in 0 .. pivot_i {
        //     // If the bit in the pivot column is set, add the pivot row to this one
        //     if a.get_unchecked(i + pivot_j.word*lda)[pivot_j.lane].bitand(col_mask) != 0 {
        //         a.get_unchecked_mut(i + j*lda).bitxor_assign(pivot_row_data);
        //     }        
        // }
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
        let n = 1024;
        let rows = n;
        let cols = ceil_div(ceil_div(n, WORD_BITS), SIMD_LANES);

        let mut rng = ChaCha8Rng::seed_from_u64(0xeface14cd35a75b5);

        let mut data : Vec<SimdWord> = vec![Simd::splat(0); rows*cols];
        for a in data.iter_mut(){
            for i in 0usize .. SIMD_LANES {
                (*a)[i] = rng.gen::<Word>();
            }
        }

        bench.iter(|| unsafe {
                _row_reduce(&mut data, rows, cols-1, WORD_BITS)
            }
        );
        assert!(data.iter().map(|x| x.reduce_sum()).sum::<Word>() > 0);
    }
}
