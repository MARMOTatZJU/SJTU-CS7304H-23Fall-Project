"""
FAILED
NOTE: QDA can be extremely impracical for high-dimensional data
    as determinant computing/inversion computing/eigenvalue computing
        are difficult in such case.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .utils import compute_pseudo_determinant


EPSILON = 1e-1


class QDAClassiffier(BaseEstimator, ClassifierMixin):

    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.means = None
        self.covs = None
        self.detcovs = None
        self.priors = None
    
    def fit(self, X : np.ndarray, y : np.ndarray):
        N,D = X.shape
        C = self.num_classes
        self.means = [None for _ in range(C)]
        self.covs = [None for _ in range(C)]
        self.detcovs = [None for _ in range(C)]
        self.priors = [None for _ in range(C)]

        for i_class in range(C):
            print(i_class)
            in_class_sample_mask = (y == i_class)
            in_class_sample_idxs = np.argwhere(in_class_sample_mask).reshape(-1)

            in_class_X = np.take(X, in_class_sample_idxs, axis=0)
            in_class_mean = in_class_X.mean(axis=0)
            in_class_X_centered = in_class_X - in_class_mean

            in_class_cov = in_class_X_centered.T @ in_class_X_centered
            in_class_cov += EPSILON * np.eye(D)  # make it invertible
            # res = np.linalg.solve(in_class_cov, in_class_X_centered[0])
            in_class_detcov = np.linalg.det(in_class_cov)
            print(in_class_detcov)
            in_class_prior = np.array((len(in_class_X) / N,))

            self.means[i_class] = in_class_mean.reshape(1, D)
            self.covs[i_class] = in_class_cov.reshape(1, D, D)
            self.detcovs[i_class] = in_class_detcov.reshape(1,)
            self.priors[i_class] = in_class_prior.reshape(1,)

        return self

    # def predict(self, X : np.ndarray) -> np.ndarray:
    #     N,D = X.shape
    #     C = self.num_classes
    #     all_samples_all_classes_confidences = list()

    #     for i_class in range(C):
    #         print(i_class)
    #         mean = self.means[i_class]
    #         conv = self.covs[i_class]
    #         detcov = self.detcovs[i_class]
    #         prior = self.priors[i_class]

    #         X_centered = X - mean  # shape=(N, D)

    #         # shape=(N,)
    #         confidence = \
    #             - 1 / 2 * X_centered.T @ conv @ X_centered- \
    #             - 1 / 2 * np.log(detcov) + \
    #             np.log(prior)

    #         all_samples_all_classes_confidences.append(confidence)

    #     # shape=(N, C)
    #     all_samples_all_classes_confidences = \
    #         np.stack(all_samples_all_classes_confidences, axis=1)

    #     # shape=(N,)
    #     class_prediction_results = \
    #         np.argmax(all_samples_all_classes_confidences, axis=1)

    #     return class_prediction_results
            





