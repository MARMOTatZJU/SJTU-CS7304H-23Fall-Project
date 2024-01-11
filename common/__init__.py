from .qda import QDAClassiffier
from .svm import SVMClassiffier, MultiClassSVMClassiffier
from .mlp import MultiLayerPerceptronClassifier

__all__ = [
    'QDAClassiffier',
    'SVMClassiffier',
    'MultiClassSVMClassiffier',
    'MultiLayerPerceptronClassifier',
]
