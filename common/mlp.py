from typing import List, Optional, Callable
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torchvision


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    NOTE: imported from `torchvision.ops.MLP` in 0.16.1, a future version

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
        # _log_api_usage_once(self)  # NOTE: available only in future version



class MultiLayerPerceptronClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            num_classes,
            verbose=False,
            n_layers=5,
            n_epochs=30,
            batch_size=1024,
            optim_type='Adam',
            optim_lr=1e-3,
            optim_weight_decay=1e-4,
            ):
        self.num_classes = num_classes
        self.verbose = verbose
        self.model = None
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optim_type = optim_type
        self.optim_lr = optim_lr
        self.optim_weight_decay = optim_weight_decay


    def fit(self, X : np.ndarray, y : np.ndarray):
        device = self.get_device()
        optim_config = dict(
            lr=self.optim_lr,
            weight_decay=self.optim_weight_decay,
        )

        N, D = X.shape
        C = self.num_classes
        base_channels = int(np.floor(np.log2(D)))

        hidden_channels=[
            int(2**(base_channels - i_layer))
            for i_layer in range(self.n_layers)
            ] + [C,]
        self.model = MLP(
            in_channels=int(D),
            hidden_channels=hidden_channels,
        ).to(device)
        # print(self.model)

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X).type(torch.float32),
            torch.from_numpy(y).type(torch.int64),
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            )
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = getattr(torch.optim, self.optim_type)(
            self.model.parameters(),
            **optim_config
        )

        self.model.train()
        for i_epoch in range(self.n_epochs):
            for i_iteration, data in enumerate(data_loader):
                optimizer.zero_grad()
                batch_X, batch_y = data
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                if self.verbose:
                    print(f'{i_epoch}/{i_iteration}, loss: {loss.detach().cpu().numpy()}')
        self.model.eval()

    def predict_proba(self, X : np.ndarray) -> np.ndarray:
        device = self.get_device()
        self.model.to(device)
        X_tensor = torch.from_numpy(X).type(torch.float32).to(device)
        self.model.eval()
        logits = self.model(X_tensor)
        proba = torch.nn.functional.softmax(logits, dim=1)
        proba = proba.detach().cpu().numpy()

        return proba

    def predict(self, X : np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        all_samples_class_ids = np.argmax(proba, axis=1)

        return all_samples_class_ids

    def get_device(self,):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device






