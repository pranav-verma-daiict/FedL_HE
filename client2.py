import requests
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Define a simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load Iris dataset
iris = load_iris()
X_train, X_val, y_train, y_val = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Create DataLoader
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model on client2's data
def train():
    model.train()
    for epoch in range(1):  # Single epoch for simplicity
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

# Evaluate the model on the validation set
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Convert model parameters to a single NumPy array
def get_params_as_array(model):
    params = []
    for param in model.parameters():
        params.append(param.data.cpu().numpy().astype(np.float32).flatten())  # Ensure float32
    return np.concatenate(params)

# Set model parameters from a single NumPy array
def set_params_from_array(model, params_array):
    offset = 0
    for param in model.parameters():
        param_length = np.prod(param.size())
        param_shape = param.size()
        new_param = params_array[offset:offset + param_length].reshape(param_shape).astype(np.float32)  # Ensure float32
        param.data = torch.tensor(new_param).to(param.device)
        offset += param_length

# Main function
if __name__ == "__main__":
    epoch = 0
    while True:
        epoch += 1
        # Train and evaluate before sending parameters to server
        train()
        accuracy_before = evaluate()
        print(f"Client2 - Epoch {epoch}: Accuracy before update: {accuracy_before:.2f}%")

        params_array = get_params_as_array(model)
        response = requests.post('http://127.0.0.1:5000/update', data=pickle.dumps(params_array))
        updated_params_array = np.array(pickle.loads(response.content), dtype=np.float32)  # Ensure float32

        # Update local model with received parameters
        set_params_from_array(model, updated_params_array)

        # Evaluate after receiving updated parameters
        accuracy_after = evaluate()
        print(f"Client2 - Epoch {epoch}: Accuracy after update: {accuracy_after:.2f}%")

        if accuracy_after >= 80.0:
            print(f"Client2 achieved an accuracy of {accuracy_after:.2f}% in epoch {epoch}. Stopping training.")
            break
