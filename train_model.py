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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
epochs = 500
loss_history = []
#%%
for epoch in range(epochs):
    model.train()
    _loss_history = []
    _validation_loss_history = []
    for batch in train_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = mse_loss(logits, y)
        loss.backward()
        optimizer.step()
        _loss_history.append(loss.item())
    if epoch % 25 == 0:
        print(f"Epoch {epoch}, Loss: {np.mean(_loss_history)}")
        loss_history.append(np.mean(_loss_history))
        average_loss = 0.0
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                logits = model(x)
                val_loss = mse_loss(logits, y)
                average_loss += val_loss.item()
        average_loss /= len(val_loader)
        _validation_loss_history.append(average_loss)
        print(f"Validation Loss: {average_loss}")

import matplotlib.pyplot as plt

# Plot the loss history over time
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss History")
plt.legend()
plt.grid(True)
plt.show()

#%%
# Calculate accuracy on the training set
model.eval()
correct_predictions = 0
total_predictions = 0
MAE = []
with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        predictions = model(x)
        # Calculate Mean Absolute Error (MAE)
        MAE.append(torch.mean(torch.abs(predictions - y)).item())

        # predicted_labels = predictions.round()  # Round predictions to nearest integer
        # correct_predictions += (predicted_labels == y).all(dim=1).sum().item()
        # total_predictions += y.size(0)

average_MAE = sum(MAE) / len(MAE)
print(f"Average MAE on training set: {average_MAE}")
print(f'Precision: {100.00 - (average_MAE * 100.00):.2f}%')
#%%
# Save the trained model
model_save_path = f"models/trained_model_{average_MAE}.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

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