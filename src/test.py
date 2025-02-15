import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, output_size=1):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

# Function to generate dataset given a function func
def generate_data(func, num_samples=1000, input_size=10):
    x = np.random.randn(num_samples, input_size).astype(np.float32)
    y = np.array([func(xi) for xi in x], dtype=np.float32).reshape(-1, 1)
    return torch.tensor(x), torch.tensor(y)

# Training function
def train_nn(model, func, num_epochs=10, batch_size=1000, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        x_train, y_train = generate_data(func, num_samples=batch_size)
        
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Example usage
def my_function(x):
    return x[3]
    #return np.sin(x.sum())  # Example: sum inputs and apply sine

# Initialize model
model = SimpleNN()

# Train the model
train_nn(model, my_function)

model.eval()
x, y = generate_data(my_function, num_samples=10)
yhat = model(x)

print(1000*y, 1000*yhat)

