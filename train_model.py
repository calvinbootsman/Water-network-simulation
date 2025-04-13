#%%
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import random
# Load the CSV file into a pandas DataFrame
file_path = 'simulation_results.csv'
simulation_results = pd.read_csv(file_path)
print(simulation_results.head())

#%%
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class SimulationDataset(torch.utils.data.Dataset):
    def __init__(self, file_location='simulation_results.csv'):
        simulation_results = pd.read_csv(file_path)
        dataset = simulation_results[['flow_35', 'flow_33', 'flow_5', 'flow_18', 'pressure_22', 'pressure_15', 'pressure_4', 'pressure_3', 'flow_9', 'pressure_8']].values

        self.data = torch.tensor(dataset, dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, :8]
        y = self.data[idx, 8:]
        return x, y

# %%
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
#%%
# Split the dataset into training, validation, and test sets
dataset = SimulationDataset()
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#%%
model = NeuralNetwork().to(device)
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 50

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = mse_loss(logits, y)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        
        average_loss = 0.0
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                logits = model(x)
                val_loss = mse_loss(logits, y)
                average_loss += val_loss.item()
        average_loss /= len(val_loader)
        print(f"Validation Loss: {average_loss}")

# %%
input_dataset = simulation_results[['flow_35', 'flow_33', 'flow_5', 'flow_18', 'pressure_22', 'pressure_15', 'pressure_4', 'pressure_3']].values
output_dataset = simulation_results[['flow_9', 'pressure_8']].values
input_tensor = torch.tensor(input_dataset, dtype=torch.float32, device=device)
output_tensor = torch.tensor(output_dataset, dtype=torch.float32, device=device)

model.eval()
# Sample a few random items from the dataset

sample_indices = random.sample(range(len(input_tensor)), 5)
sample_inputs = input_tensor[sample_indices]
sample_outputs = output_tensor[sample_indices]

print("Sample Inputs:", sample_inputs)
print("Sample Outputs:", sample_outputs)
with torch.no_grad():
    predictions = model(input_tensor)
    predicted_outputs = predictions[sample_indices]
    print("Predicted Outputs:", predicted_outputs)