import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torchvision


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
        self.model = torchvision.ops.MLP(
            in_channels=int(D),
            hidden_channels=hidden_channels,
        ).to(device)
        # print(self.model)

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X).type(torch.float32).to(device),
            torch.from_numpy(y).type(torch.int64).to(device),
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






