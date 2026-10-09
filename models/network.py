import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, dropout_rate = 0.0):

        super().__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.layer2 = nn.Linear(hidden_dim, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.output_layer = nn.Linear(32, output_dim)


    def forward(self, x):

        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.output_layer(x)

        return x

    
if __name__ == "__main__":

    model = SimpleMLP()
    print("MODEL ARCHITECTURE: ")
    print(model)

    dummy_data = torch.randn(5, 2)

    predictions = model(dummy_data)

    print("\n Predictions Shape: ", predictions.shape)