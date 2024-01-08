import os

import numpy as np
import pandas as pd


def compute_pseudo_determinant(mat):
    print('compute_pseudo_determinant')
    eig_result = np.linalg.eig(mat)
    pdet = np.product(eig_result.eigenvalues[eig_result.eigenvalues > 1e-12])

    return pdet

def argmedian(x):
    return np.argpartition(x, len(x) // 2)[len(x) // 2]


def export_result_for_submission(predicted_labels : np.array, path : str):
    export_df = pd.DataFrame(
        data=predicted_labels.astype(np.int64),
        columns=('label',),
        )
    export_df.index.name = 'ID'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    export_df.to_csv(
        path,
        header=True,
        index=True,
        )
    print(f'Result exported to: {path}.')
