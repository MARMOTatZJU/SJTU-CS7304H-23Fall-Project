import argparse
import os
import random
from copy import deepcopy
import pickle

import numpy as np
from common import SVMClassiffier, MultiClassSVMClassiffier, MultiLayerPerceptronClassifier
from common.utils import export_result_for_submission


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp')
    args = parser.parse_args()

    return args


def main(args):
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

    sampled_indexes = list(range(N))


    if args.model == 'svm-h':
        classifier = MultiClassSVMClassiffier(num_classes=20)
    elif args.model == 'svm-s':
        classifier = MultiClassSVMClassiffier(num_classes=20, slack_coeff=1e+0)
    elif args.model == 'mlp':
        classifier = MultiLayerPerceptronClassifier(num_classes=20, verbose=True)
    elif args.model == 'mlp-with-model-selection':
        classifier = MultiLayerPerceptronClassifier(num_classes=20, verbose=True, **{"optim_weight_decay": 1e-06, "optim_lr": 0.0001, "n_layers": 1, "n_epochs": 23, "batch_size": 128})
    else:
        raise ValueError(f'Unsupported model: {args.model}')

    classifier.fit(data[sampled_indexes], labels[sampled_indexes])
    predicted_labels = classifier.predict(data)

    # DEPRECATED:
    # def compute_gaussian_kernel_distance(x, y, sigma=0.01):
    #     return np.exp(- np.linalg.norm(x-y)**2 / 2 / (sigma**2))
    # classifier = MultiClassSVMClassiffier(num_classes=20, inner_prod_func=compute_gaussian_kernel_distance)

    correctness_mask = (predicted_labels == labels)
    print(sum(correctness_mask)/len(correctness_mask))

    export_dir = f'./{args.model}'
    os.makedirs(export_dir, exist_ok=True)

    predicted_test_labels = classifier.predict(test_data)
    export_path = f'./{export_dir}/submission/.csv'
    export_result_for_submission(predicted_test_labels, export_path)

    classifier_path = f'./{export_dir}/classifier.pkl'
    with open(classifier_path, 'wb') as f:
        pickle.dump(classifier, f)
    print(f'Classifier exported at: {classifier_path}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
