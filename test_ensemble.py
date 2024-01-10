"""
Model ensemble
"""
import os
import random
from copy import deepcopy
import pickle
import json
import datetime
import glob

import numpy as np
import sklearn
from sklearn.model_selection import RandomizedSearchCV


from common import MultiLayerPerceptronClassifier
from common.utils import export_result_for_submission


p = 'data/train_feature.pkl'
with open(p, 'rb') as f:
    data = pickle.load(f)
data = data.toarray()

p = 'data/train_labels.npy'
with open(p, 'rb') as f:
    labels = np.load(f)

p = 'data/test_feature.pkl'
with open(p, 'rb') as f:
    test_data = pickle.load(f)
test_data = test_data.toarray()

num_classes = len(np.unique(labels))
N = len(test_data)
C = num_classes

save_dir = './model_selection_results'
model_selector_paths = glob.glob(f'{save_dir}/*/model_selector.pkl')
M = len(model_selector_paths)

model_selectors = list()
for p in model_selector_paths:
    with open(p, 'rb') as f:
        model_selector = pickle.load(f)
        # model_selector.best_estimator_.set_device('cpu')
    model_selectors.append(model_selector)

all_models_predicted_test_logits = list()
all_models_predicted_test_probas = list()
for model_selector in model_selectors:
    predicted_test_logits = model_selector.best_estimator_.predict_logit(test_data)  # shape=(N, C)
    predicted_test_probas = model_selector.best_estimator_.predict_proba(test_data)  # shape=(N, C)
    all_models_predicted_test_logits.append(predicted_test_logits)
    all_models_predicted_test_probas.append(predicted_test_probas)

# ensemble
all_models_predicted_test_logits = np.stack(all_models_predicted_test_logits, axis=0)  # shape=(M, N, C)
all_models_predicted_test_probas = np.stack(all_models_predicted_test_probas, axis=0)  # shape=(M, N, C)
ensemble_predicted_test_logits = all_models_predicted_test_logits.mean(axis=0)  # shape=(N, C)
ensemble_predicted_test_probas = all_models_predicted_test_probas.mean(axis=0)  # shape=(N, C)

ensemble_predicted_test_labels_with_probas = np.argmax(ensemble_predicted_test_probas, axis=1)  # shape(N,), dtype=np.int64
ensemble_predicted_test_labels_with_logits = np.argmax(ensemble_predicted_test_logits, axis=1)  # shape(N,), dtype=np.int64

ensemble_predicted_test_labels = ensemble_predicted_test_labels_with_probas
export_path = f'./ensemble_submission.csv'
export_result_for_submission(ensemble_predicted_test_labels, export_path)

print(len(ensemble_predicted_test_labels_with_probas == ensemble_predicted_test_labels_with_logits)/N)
for i_model, proba in enumerate(all_models_predicted_test_probas):
    predicted_test_labels_with_probas = np.argmax(proba, axis=1)  # shape=(N,)
    diff = len(ensemble_predicted_test_labels_with_probas == predicted_test_labels_with_probas)/N
    print(f'model {i_model}, diff={diff}')

# from IPython import embed;embed();exit()
