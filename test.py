import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import numpy as np

def run():

    # Generate some sample data
    np.random.seed(42)
    X_train = np.random.rand(100, 1).astype(np.float32)
    y_train = 2 * X_train + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

    X_val = np.random.rand(20, 1).astype(np.float32)
    y_val = 2 * X_val + 1 + 0.1 * np.random.randn(20, 1).astype(np.float32)

    X_test = np.random.rand(20, 1).astype(np.float32)
    y_test = 2 * X_test + 1 + 0.1 * np.random.randn(20, 1).astype(np.float32)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)

    X_val_tensor = torch.from_numpy(X_val)
    y_val_tensor = torch.from_numpy(y_val)

    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)

    # Create PyTorch datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Define a simple regression model
    class RegressionModel(pl.LightningModule):
        def __init__(self):
            super(RegressionModel, self).__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_pred = self(x)
            loss = nn.MSELoss()(y_pred, y)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_pred = self(x)
            loss = nn.MSELoss()(y_pred, y)
            return loss

        def test_step(self, batch, batch_idx):
            x, y = batch
            y_pred = self(x)
            loss = nn.MSELoss()(y_pred, y)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    # Create Lightning Trainer
    trainer = pl.Trainer(max_epochs=10)

    # Create model and train
    model = RegressionModel()
    trainer.fit(model, DataLoader(train_dataset, batch_size=32), DataLoader(val_dataset, batch_size=32))

    # Test the trained model
    result = trainer.test(model, DataLoader(test_dataset, batch_size=32))
    print(result)

if __name__ == '__main__':
    run()
