import numpy as np


def compute_pseudo_determinant(mat):
    print('compute_pseudo_determinant')
    eig_result = np.linalg.eig(mat)
    pdet = np.product(eig_result.eigenvalues[eig_result.eigenvalues > 1e-12])

    return pdet

def argmedian(x):
    return np.argpartition(x, len(x) // 2)[len(x) // 2]
