import numpy as np
import random
import tenseal as ts
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data for clients
data_splits = []
for _ in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    data_splits.append((X_train, X_test, y_train, y_test))

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# Client class
class Client:
    def __init__(self, data):
        self.X_train, self.X_test, self.y_train, self.y_test = data
        self.model = SimpleModel(self.X_train.shape[1], len(np.unique(self.y_train)))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        self.context.generate_galois_keys()
        self.context.global_scale = 2**40
        self.train_model()

    def train_model(self, epochs=10):
        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(torch.tensor(self.X_train, dtype=torch.float32))
            loss = self.criterion(outputs, torch.tensor(self.y_train, dtype=torch.long))
            loss.backward()
            self.optimizer.step()
        # we need the key generation here itself as we want each client to use the SAME keys.
        # ps: need to work on it later as its not safe to use same key by all clients for privacy reasons

    # the params should be in arrray/vector to be encrypted
    def get_params_as_array(self):
        params = []
        for param in self.model.parameters():
            params.append(param.data.view(-1).numpy())
        return np.concatenate(params)

    # update the params of the model with the NEWLY received values
    def set_params_from_array(self, params_array):
        params_array = np.array(params_array)  # Ensure params_array is a NumPy array
        offset = 0
        for param in self.model.parameters():
            param_length = param.numel()
            param_shape = param.size()
            new_param = params_array[offset:offset + param_length].reshape(param_shape)
            param.data = torch.tensor(new_param, dtype=torch.float32)
            offset += param_length

    # before sending params to the server
    def encrypt_params(self):
        params_array = self.get_params_as_array()
        enc_params = ts.ckks_vector(self.context, params_array)
        return enc_params

    # upon receiving params from the server, as the local training is done over plaintexts by the client
    def decrypt_params(self, enc_params):
        params_array = enc_params.decrypt()
        self.set_params_from_array(params_array)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.tensor(self.X_test, dtype=torch.float32))
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == torch.tensor(self.y_test, dtype=torch.long)).float().mean().item()
        return accuracy

# Initialize clients
clients = [Client(data_splits[i]) for i in range(10)]

# Server function
def federated_learning_server(clients):
    while True:
        # Select client[0] to send its parameters. It could be any random user as well, it would mean the same.
        enc_params = clients[0].encrypt_params()

        # Send encrypted parameters to 3 randomly chosen clients
        selected_clients = random.sample(clients[1:], 3)
        for client in selected_clients:
            client.decrypt_params(enc_params)
            client.train_model()
            accuracy = client.evaluate()
            print(f"Client accuracy after update: {accuracy:.2f}")
            if accuracy >= 0.85:
                # Send final parameters to all clients
                final_enc_params = client.encrypt_params()
                for c in clients:
                    c.decrypt_params(final_enc_params)
                    c.train_model()
                    accuracy = c.evaluate()
                    print(f"Client final accuracy: {accuracy:.2f}")
                    print(f"Client final parameters: {c.get_params_as_array()}")
                return
            enc_params = client.encrypt_params()
            # want to make sure the three users have sufficient accuracy to finalize the param values

        # Compute average of the updated encrypted parameters
        enc_avg_params = selected_clients[0].encrypt_params()
        for client in selected_clients[1:]:
            enc_avg_params += client.encrypt_params()

        # Use internal division method of CKKSVector
        enc_avg_params = enc_avg_params * (1/3)

        # Send averaged encrypted parameters to another 3 randomly chosen clients (hopefully new clients)
        next_selected_clients = random.sample(clients, 3)
        for client in next_selected_clients:
            client.decrypt_params(enc_avg_params)
            client.train_model()
            accuracy = client.evaluate()
            print(f"Next client accuracy after update: {accuracy:.2f}")
            if accuracy < 0.85:
                break
        else:
            # Send final parameters to all clients
            for client in clients:
                client.decrypt_params(enc_avg_params)
                client.train_model()
                accuracy = client.evaluate()
                print(f"Client final accuracy: {accuracy:.2f}")
                print(f"Client final parameters: {client.get_params_as_array()}")
            return

# Run federated learning
Stime = time.time()
federated_learning_server(clients)
Etime= time.time()
print("Total runtime: ", Etime - Stime)
