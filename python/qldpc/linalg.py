from .qecc_util import GF2
import numpy as np
from typing import List
import tempfile
import subprocess

_default_sage_cutoff = 512**3

def exec_sage(A : np.array, sage_cmd : str) -> np.array:
    with tempfile.NamedTemporaryFile(mode='w+b') as datafile:
        # Save matrix to a tempfile
        np.save(datafile, A, allow_pickle=False)
        datafile.flush()
        # Execute sage commands
        sage_script = f'''
        import numpy as np
        A = matrix(GF(2), np.load({datafile.name}, allow_pickle=False))
        {sage_cmd}
        np.save({datafile.name}, A.numpy(), allow_pickle=False)
        '''
        completedProcess = subprocess.run(['sage', '-c', ';'.join(s.strip() for s in sage_script.splitlines())], capture_output=True)
        if completedProcess.returncode != 0:
            raise RuntimeError(completedProcess)
        # Read back result
        datafile.seek(0)
        result = np.load(datafile, allow_pickle=False)
    return result

def gf2_null_space(A : np.array) -> np.array:
    return GF2(A).null_space()

def gf2_column_space(A : np.array) -> np.array:
    return GF2(A).column_space()

def gf2_row_reduce(A : np.array, ncols : int = None, use_sage = False, sage_cutoff = None) -> np.array:
    if sage_cutoff is None:
        sage_cutoff = _default_sage_cutoff
    if use_sage and A.shape[0]*A.shape[1]**2 >= sage_cutoff and ncols is None:
        return exec_sage(A, f'A = A.echelonize()')
    else:
        return GF2(A).row_reduce(ncols=ncols)

def gf2_get_pivots(A : np.array) -> List[int]:
    largest_index = (A!=0).argmax(axis=1)
    return np.extract(A[range(A.shape[0]), largest_index]!=0, largest_index)

def gf2_matrix_rank(A : np.array) -> int:
    return get_rank(A)

def gf2_solve(A : np.array, b : np.array) -> np.array:
    pass

def get_rank(A : np.array) -> int:
    return np.linalg.matrix_rank(GF2(A))

