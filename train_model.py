#%%
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import random
import numpy as np
# Load the CSV file into a pandas DataFrame
directory = 'dataset_output'
file_name = 'good_dataset_20250421_174438.csv'
file_path = f"{directory}/{file_name}"
simulation_results = pd.read_csv(file_path)
# print(simulation_results.head())

#%%
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class SimulationDataset(torch.utils.data.Dataset):
    def __init__(self, file_location='simulation_results.csv', normalize=True):
        simulation_results = pd.read_csv(file_location)
        dataset = simulation_results[['flow_35', 'flow_33', 'flow_5', 'flow_18', 'pressure_22', 'pressure_15', 'pressure_4', 'pressure_3', 'flow_9', 'pressure_8']].values # should also add node_27
        # Remove rows with NaN values
        dataset = dataset[~pd.isna(dataset).any(axis=1)]
        self.original_dataset = dataset.copy()
        self.min_values = dataset.min(axis=0)
        self.max_values = dataset.max(axis=0)
        if normalize:
            # Normalize the dataset
            dataset = (dataset - self.min_values) / (self.max_values - self.min_values)
        self.data = torch.tensor(dataset, dtype=torch.float32, device=device)

    def denormalize(self, dataset):
        # Denormalize the dataset to the original range
        dataset = dataset * (self.max_values - self.min_values) + self.min_values
        return dataset
    
    def denormalize_output(self, data):
        # Denormalize the output data to the original range
        _max_values = torch.tensor(self.max_values, dtype=torch.float32, device=device)
        _min_values = torch.tensor(self.min_values, dtype=torch.float32, device=device)
        data = data * (_max_values[8:] - _min_values[8:]) + _min_values[8:]
        return data
    
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
dataset = SimulationDataset(file_path)
train_size = int(0.003 * len(dataset))
val_size = int(0.0005 * len(dataset))
test_size = len(dataset) - train_size - val_size

print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

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
epochs = 5000
loss_history = []

max_no_improvement = 6
no_improvement_count = 0
best_val_loss = float('inf')
early_stopping = False
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
        if average_loss < best_val_loss:
            best_val_loss = average_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                print(f"Early stopping at epoch {epoch}")
                early_stopping = True
                break
    if early_stopping:
        print(f"Early stopping triggered at epoch {epoch}")
        break
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
        if torch.any(torch.isnan(x)) or torch.any(torch.isnan(y)):
            print("Encountered NaN values in the dataset.")
            # continue
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

#%%
model.eval()
num_samples_to_show = 5

# Get random indices from the full dataset
# Ensure indices are unique and within the bounds of the dataset
if len(dataset) >= num_samples_to_show:
    random_indices = random.sample(range(len(dataset)), num_samples_to_show)
else:
    print(f"Warning: Requested {num_samples_to_show} samples, but dataset only has {len(dataset)} entries. Showing all.")
    random_indices = list(range(len(dataset)))

print(f"\n--- Comparing {num_samples_to_show} Random Samples ---")

with torch.no_grad():
    for i, idx in enumerate(random_indices):
        x_norm, y_norm = dataset[idx]
        x_norm_batch = x_norm.unsqueeze(0)

        prediction_norm = model(x_norm_batch) # Shape is [1, 2]
        prediction_denorm = dataset.denormalize_output(prediction_norm)
        actual_denorm = dataset.denormalize_output(y_norm.unsqueeze(0)) # Use unsqueeze for potential consistency
        original_input_features = dataset.original_dataset[idx, :8]

        print(f"\nSample #{i+1} (Dataset Index: {idx})")
        # Use numpy for potentially cleaner printing of arrays
        print(f"  Input Features (Original): {np.array2string(original_input_features, precision=4, suppress_small=True)}")
        # Move tensors to CPU and convert to numpy for printing
        print(f"  Predicted Output (Denormalized): {np.array2string(prediction_denorm.cpu().numpy(), precision=4, suppress_small=True)}")
        print(f"  Actual Output    (Denormalized): {np.array2string(actual_denorm.cpu().numpy(), precision=4, suppress_small=True)}")

print("\n--------------------------------------")
