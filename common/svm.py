from itertools import product
from copy import deepcopy
import math

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from cvxopt import matrix, solvers
from scipy.stats import logistic


from .constant import EPSILON
from .utils import argmedian


class SVMClassiffier(BaseEstimator, ClassifierMixin):

    def __init__(self, slack_coeff : float=0.0, inner_prod_func=np.inner):
        self.vec_w = None
        self.scalar_b = None
        self.slack_coeff = slack_coeff
        self.support_vecs = None
        self.support_alphas = None
        self.support_labels = None
        self.inner_prod_func = inner_prod_func

    def fit(self, X : np.ndarray, y : np.ndarray):
        N, D = X.shape

        # objective
        if self.inner_prod_func is np.inner:
            mat_q_outer_vec = y.reshape(N, 1)*X  # shape=(N, D)
            MAT_Q_NP = mat_q_outer_vec @ mat_q_outer_vec.T  # shape=(N, N)
        else:
            MAT_Q_NP  = np.zeros((N, N))
            for i, j in product(range(N), range(N)):
                MAT_Q_NP[i, j] = y[i] * y[j] * self.inner_prod_func(X[i], X[j])
        MAT_Q = matrix(MAT_Q_NP)
        VEC_P = matrix(-np.ones((N,)))
        # inequality
        if self.slack_coeff > 0.0:
            # soft-margin, 0 <= alpha <= C, 2*N constraint
            MAT_G_NP = np.concatenate((
                -np.eye(N),
                np.eye(N),
            ), axis=0)
            VEC_H_NP = np.concatenate((
                -np.zeros((N,)),
                self.slack_coeff*np.ones((N,)),
            ), axis=0)
        else:
            # hard-margin, alpha >= 0, N constraint
            MAT_G_NP = -np.eye(N)
            VEC_H_NP = -np.zeros((N,))
        MAT_G = matrix(MAT_G_NP)
        VEC_H = matrix(VEC_H_NP)
        # equality
        MAT_A = matrix(y.reshape(1, N).astype(np.float64))  # shape=(1, N)
        VEC_B = matrix(np.zeros((1,)))  # shape=(1, 1)
        # solve
        result = solvers.qp(MAT_Q, VEC_P, MAT_G, VEC_H, MAT_A, VEC_B)
        # dual variables
        alphas = np.array(result['x']).reshape(-1)  # shape=(N,)
        nonzero_alpha_mask = (alphas > EPSILON).reshape(-1)  # shape=(N,)
        nonzero_idxs = np.argwhere(nonzero_alpha_mask).reshape(-1)
        # w
        if self.inner_prod_func is np.inner:
            self.vec_w = \
                (alphas[nonzero_alpha_mask].reshape(-1, 1) * y[nonzero_alpha_mask].reshape(-1, 1) * X[nonzero_alpha_mask]) \
                    .sum(axis=0)  # shape=(D,), summation on sample dimention
        else:
            self.support_vecs = X[nonzero_idxs]
            self.support_alphas = alphas[nonzero_idxs]
            self.support_labels = y[nonzero_idxs]
        # b
        scalar_b_candidates = list()
        for nonzero_idx in nonzero_idxs:
            support_vec = X[nonzero_idx]
            if self.inner_prod_func is np.inner:
                scalar_b = y[nonzero_idx] - np.inner(self.vec_w, support_vec)   # shape=(1,)
            else:
                scalar_b = y[nonzero_idx] - sum([
                    alphas[i_support] * y[i_support] * self.inner_prod_func(X[i_support], support_vec)
                    for i_support in nonzero_idxs
                ])
            scalar_b_candidates.append(scalar_b)
        selected_nonzero_idx = argmedian(scalar_b_candidates)        
        self.scalar_b = scalar_b_candidates[selected_nonzero_idx]  # shape=(1,)

    def predict_proba(self, X : np.ndarray) -> np.ndarray:
        """
        Return
            Predicted SVM labels, 
        """
        N, D = X.shape
        if self.inner_prod_func is np.inner:
            deicision_vars = np.array([
                X[i_sample] @ self.vec_w + self.scalar_b
                for i_sample in range(N)
            ]).reshape(N,)
        else:
            deicision_vars = list()
            for i_sample in range(N):
                deicision_var = self.scalar_b + sum([ alpha * label * self.inner_prod_func(vec, X[i_sample])
                    for alpha, label, vec in \
                        zip(self.support_alphas, self.support_labels, self.support_vecs)
                ])
                deicision_vars.append(deicision_var)
            deicision_vars = np.array(deicision_vars).reshape(N,)
        proba = logistic.cdf(deicision_vars)

        return proba

    def predict(self, X : np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        result = 2 * np.array(proba > 0.5, dtype=np.float64) - 1

        return result

 

class MultiClassSVMClassiffier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_classes, slack_coeff : float=0.0, inner_prod_func=np.inner):
        self.num_classes = num_classes
        self.slack_coeff = slack_coeff
        self.svms = [
            SVMClassiffier(
                slack_coeff=slack_coeff,
                inner_prod_func=inner_prod_func,
                )
            for _ in range(num_classes)
            ]
    
    def fit(self, X : np.ndarray, y : np.ndarray):
        for class_id in range(self.num_classes):
            svm_labels = deepcopy(y)
            svm_labels[y == class_id] = +1.0
            svm_labels[y != class_id] = -1.0
            classifier = self.svms[class_id]
            classifier.fit(X, svm_labels)

    def predict(self, X : np.ndarray) -> np.ndarray:
        C = self.num_classes
        all_class_probas = list()
        for class_id in range(C):
            in_class_probas = self.svms[class_id].predict_proba(X)
            all_class_probas.append(in_class_probas)
        all_class_probas = np.stack(all_class_probas, axis=1)  # shape=(N, C)
        all_samples_class_ids = all_class_probas.argmax(axis=1)

        return all_samples_class_ids
