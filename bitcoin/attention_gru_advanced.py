import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from itertools import product
import random
import os

from models.attention_gru import AttentionGRU
from losses.hybrid_loss import HybridLoss

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

#############
# Parameters
#############
csv_file = "datasets/BTC-USD.csv"
seq_len = 30
batch_size = 32
input_size = 5
output_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
# Data Loading and Preprocessing
#################################

df = pd.read_csv(csv_file)
df = df.sort_values('Date')

features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

# Split before scaling to avoid lookahead bias
train_size = int(0.8 * len(features))
train_data = features[:train_size]
test_data = features[train_size:]

# Fit the scaler on training data only
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data)

# Apply the same scaler to test data
scaled_test_data = scaler.transform(test_data)

# Create sequences for training
X_train, Y_train = [], []
for i in range(len(scaled_train_data) - seq_len):
    X_train.append(scaled_train_data[i:i+seq_len, :])
    Y_train.append(scaled_train_data[i+seq_len, 3])  # predicting the 'Close' price

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Create sequences for testing
X_test, Y_test = [], []
for i in range(len(scaled_test_data) - seq_len):
    X_test.append(scaled_test_data[i:i+seq_len, :])
    Y_test.append(scaled_test_data[i+seq_len, 3])

X_test = np.array(X_test)
Y_test = np.array(Y_test)

class BitcoinDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  
        y = self.Y[idx]
        context = x
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        context = torch.tensor(context, dtype=torch.float32)
        return x, context, y

train_dataset = BitcoinDataset(X_train, Y_train)
test_dataset = BitcoinDataset(X_test, Y_test)

#########################
# Utility Functions
#########################

def train_one_epoch(model, criterion, optimizer, train_loader):
    model.train()
    total_loss = 0.0
    for x_batch, context_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        context_batch = context_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs, attn_weights = model(x_batch, context_batch)
        pred = outputs[:, -1, 0]

        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate(model, test_loader, scaler):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x_batch, context_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            context_batch = context_batch.to(device)
            y_batch = y_batch.to(device)

            outputs, attn_weights = model(x_batch, context_batch)
            pred = outputs[:, -1, 0]

            predictions.append(pred.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    # Inverse scaling
    dummy = np.zeros((len(predictions), input_size))
    dummy[:,3] = predictions
    inv_predictions = scaler.inverse_transform(dummy)[:,3]

    dummy[:,3] = actuals
    inv_actuals = scaler.inverse_transform(dummy)[:,3]

    mse = np.mean((inv_predictions - inv_actuals)**2)
    return mse, inv_predictions, inv_actuals

def plot_training_loss(train_losses, title="Training Loss over Epochs", save_path="training_loss_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Training Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def plot_predictions(inv_actuals, inv_predictions, title="Predicted vs Actual Prices", save_path="predicted_vs_actual_prices.png"):
    plt.figure(figsize=(14, 8))
    plt.plot(inv_actuals, label="Actual Prices", color="blue", alpha=0.7)
    plt.plot(inv_predictions, label="Predicted Prices", color="orange", alpha=0.7)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()

#########################
# Hyperparameter Grid
#########################

param_grid = {
    'hidden_size': [64],
    'context_size': [5],
    'learning_rate': [0.0005],
    'alpha': [0.5],
    'num_epochs': [40]
}

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

best_mse = float('inf')
best_params = None
best_model_state = None
results = []

for hidden_size, context_size, learning_rate, alpha, num_epochs in product(*param_grid.values()):
    print(f"\nTesting configuration: hidden_size={hidden_size}, context_size={context_size}, "
          f"learning_rate={learning_rate}, alpha={alpha}, num_epochs={num_epochs}")

    # Initialize model and training components
    model = AttentionGRU(input_size=input_size, hidden_size=hidden_size, 
                         context_size=context_size, output_size=output_size, device=device).to(device)
    criterion = HybridLoss(alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, criterion, optimizer, train_loader)
        train_losses.append(avg_loss)

    mse, inv_predictions, inv_actuals = evaluate(model, test_loader, scaler)

    print(f"Configuration MSE: {mse:.4f}")

    # Check for best model
    if mse < best_mse:
        best_mse = mse
        best_params = {
            'hidden_size': hidden_size,
            'context_size': context_size,
            'learning_rate': learning_rate,
            'alpha': alpha,
            'num_epochs': num_epochs
        }
        best_model_state = model.state_dict().copy()
        best_train_losses = train_losses[:]
        best_inv_predictions = inv_predictions
        best_inv_actuals = inv_actuals

    results.append((hidden_size, context_size, learning_rate, alpha, num_epochs, mse))

# Print best result
print("\nBest Configuration:")
for k, v in best_params.items():
    print(f"{k}: {v}")
print(f"Best MSE: {best_mse:.4f}")

# Save best model
if best_model_state is not None:
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(best_model_state, "models/attention_gru_advanced.pth")

# Plot best model results
plot_training_loss(best_train_losses, title="Training Loss (Best Model)", save_path="best_training_loss_plot.png")
plot_predictions(best_inv_actuals, best_inv_predictions, title="Predicted vs Actual Prices (Best Model)", save_path="plots/best_predicted_vs_actual_prices_2.png")

print("All done!")
