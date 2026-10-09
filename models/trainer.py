import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, epochs = 150, lr = 0.01, weight_decay = 0.0):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):

        model.train()

        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)


        model.eval()

        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == y_batch).sum().item()
                total_val += y_batch.size(0)


        
        history['train_loss'].append(running_train_loss / len(train_loader))
        history['val_loss'].append(running_val_loss/ len(val_loader))
        history['train_acc'].append(correct_train / total_train)
        history['val_acc'].append(correct_val / total_val)

    return history



if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from data.datasets import get_train_val_loaders
    from models.network import SimpleMLP

    print("Loading data ...")
    t_loader, v_loader = get_train_val_loaders(batch_size=32)

    print("Initializing model ...")
    model = SimpleMLP()

    print("Training 5 epochs for test ...")
    history = train_model(model, t_loader, v_loader, epochs=5)

    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    print("Trainer works!")