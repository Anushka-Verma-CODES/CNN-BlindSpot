import torch
from torch.utils.data import TensorDataset, DataLoader

def generate_synthetic_data(num_samples=620):
    """Generating a noisy 2D circular dataset for classification"""

    torch.manual_seed(42)

    X = torch.rand(num_samples, 2) * 2 - 1

    radius = torch.sqrt(X[:, 0]**2 + X[:, 1]**2)

    y = (radius > 0.6).to(torch.int64)

    X += torch.randn_like(X) * 0.15

    return X, y


def get_train_val_loaders(num_train = 120, batch_size = 32):
    """Generate and Split Data"""

    X, y = generate_synthetic_data(num_samples = 620)

    X_train = X[:num_train]
    y_train = y[:num_train]

    X_val = X[num_train:]
    y_val = y[num_train:]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader


if __name__ == "__main__":
    t_loader, v_loader = get_train_val_loaders(num_train = 120, batch_size = 32)

    print("Number of Training Batches: ", len(t_loader))
    print("Number of Validation Batches: ", len(v_loader))
