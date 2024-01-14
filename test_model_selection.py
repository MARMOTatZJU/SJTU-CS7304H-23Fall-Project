"""
Model selection with:
- K-Fold Cross Validation
- Randomized Search
"""
import os
import random
from copy import deepcopy
import pickle
import json
import datetime

import numpy as np
import sklearn
from sklearn.model_selection import RandomizedSearchCV


from common import MultiLayerPerceptronClassifier
from common.utils import export_result_for_submission


def main():
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
    N = len(data)

    classifier = MultiLayerPerceptronClassifier(num_classes=num_classes)
    model_selector = RandomizedSearchCV(
        classifier,
        param_distributions=dict(
            n_layers=list(range(1, 10)),
            n_epochs=list(range(10, 30)),
            batch_size=[2**i for i in range(4, 12)],
            optim_lr=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            optim_weight_decay=[1e-4, 1e-5, 1e-6],
        ),
        n_iter=200,
        n_jobs=2,
        cv=5,
        verbose=True,
        return_train_score=True,
        refit=True,
    )
    model_selector.fit(data, labels)

    save_dir = './model_selection_results'
    cv_dirname = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    cv_dir = f'{save_dir}/{cv_dirname}'
    os.makedirs(cv_dir, exist_ok=True)

    best_estimator = model_selector.best_estimator_
    predicted_test_labels = best_estimator.predict(test_data)
    export_path = f'./{cv_dir}/submission.csv'
    export_result_for_submission(predicted_test_labels, export_path)

    export_model_selector_path = f'./{cv_dir}/model_selector.pkl'
    with open(export_model_selector_path, 'wb') as f:
        pickle.dump(model_selector, f)
    print(f'Model selector exported at: {export_model_selector_path}')
    print(f'    Useful attributes: `model_selector.cv_results_`, `model_selector.best_params_`, `model_selector.best_estimator_`, etc.')

    export_best_params_path = f'{cv_dir}/best_params.json'
    with open(export_best_params_path, 'w') as f:
        json.dump(model_selector.best_params_, f)
    print(f'Best params exported at: {export_best_params_path}')

if __name__ == '__main__':
    main()
