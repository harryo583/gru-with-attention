import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import random
import os
from itertools import product

# If you have different models or losses, adjust imports accordingly
from models.attention_gru import AttentionGRU
from losses.hybrid_loss import HybridLoss

######################################
# Set Random Seed for Reproducibility
######################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################################
# Hyperparameters and Configuration
####################################
csv_file = "datasets/BTC-USD.csv"
seq_len = 30
batch_size = 32
input_size = 5
output_size = 1

# Early stopping parameters
early_stop_patience = 5  # number of epochs with no improvement before stopping

# Number of seeds and runs per configuration to check stability
num_seeds = 3

#################################
# Data Loading and Preprocessing
#################################

df = pd.read_csv(csv_file)
df = df.sort_values('Date')
features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

# Train/validation/test split
train_size = int(0.6 * len(features))  # first 60% for training
val_size = int(0.2 * len(features))     # next 20% for validation
test_size = len(features) - train_size - val_size  # last 20% for test

train_data = features[:train_size]
val_data = features[train_size:train_size+val_size]
test_data = features[train_size+val_size:]

# Fit scaler on training data only
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data)
scaled_val_data = scaler.transform(val_data)
scaled_test_data = scaler.transform(test_data)

def create_sequences(data, seq_len=30):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, :])
        Y.append(data[i+seq_len, 3])  # predict 'Close'
    return np.array(X), np.array(Y)

X_train, Y_train = create_sequences(scaled_train_data, seq_len)
X_val, Y_val = create_sequences(scaled_val_data, seq_len)
X_test, Y_test = create_sequences(scaled_test_data, seq_len)

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

####################
# Utility Functions
####################
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
    return total_loss / len(train_loader)

def evaluate(model, loader, scaler):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x_batch, context_batch, y_batch in loader:
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
    dummy[:, 3] = predictions
    inv_predictions = scaler.inverse_transform(dummy)[:, 3]

    dummy[:, 3] = actuals
    inv_actuals = scaler.inverse_transform(dummy)[:, 3]

    mse = np.mean((inv_predictions - inv_actuals) ** 2)
    return mse, inv_predictions, inv_actuals

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

#################################
# Advanced Hyperparameter Tuning
#################################
# Instead of a fixed grid, we'll do random search
import itertools

# Parameter distributions for random search
hidden_size_choices = [32, 64, 128, 256]
context_size_choices = [5, 10, 20]
learning_rate_choices = [0.0001, 0.0005, 0.001]
alpha_choices = [0.1, 0.25, 0.5, 0.75, 0.9]
dropout_choices = [0.0, 0.1, 0.2, 0.3]
weight_decay_choices = [0.0, 1e-4, 1e-3]

num_epochs = 50
num_configurations = 10  # Randomly pick 10 configurations to try

param_candidates = list(itertools.product(hidden_size_choices, context_size_choices, learning_rate_choices,
                                          alpha_choices, dropout_choices, weight_decay_choices))
random.shuffle(param_candidates)
param_candidates = param_candidates[:num_configurations]

train_dataset = BitcoinDataset(X_train, Y_train)
val_dataset = BitcoinDataset(X_val, Y_val)
test_dataset = BitcoinDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

best_mse = float('inf')
best_params = None
best_model_state = None
best_inv_predictions = None
best_inv_actuals = None

for (hidden_size, context_size, learning_rate, alpha, dropout, weight_decay) in param_candidates:
    print(f"\nTesting configuration: hidden_size={hidden_size}, context_size={context_size}, "
          f"learning_rate={learning_rate}, alpha={alpha}, dropout={dropout}, weight_decay={weight_decay}")

    # Running multiple seeds for stability
    seed_mses = []
    for seed in range(num_seeds):
        set_seed(seed)
        model = AttentionGRU(input_size=input_size, hidden_size=hidden_size, 
                             context_size=context_size, output_size=output_size, device=device, dropout=dropout).to(device)
        criterion = HybridLoss(alpha=alpha)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=False)

        best_val_mse_for_seed = float('inf')
        best_state_for_seed = None
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, criterion, optimizer, train_loader)
            val_mse, _, _ = evaluate(model, val_loader, scaler)
            scheduler.step(val_mse)

            # Early stopping check
            if val_mse < best_val_mse_for_seed:
                best_val_mse_for_seed = val_mse
                best_state_for_seed = model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch} for seed {seed}")
                    break

        # Restore best state for this seed and evaluate on test
        model.load_state_dict(best_state_for_seed)
        test_mse, inv_predictions, inv_actuals = evaluate(model, test_loader, scaler)
        seed_mses.append(test_mse)

    mean_mse = np.mean(seed_mses)
    print(f"Mean MSE over {num_seeds} seeds: {mean_mse:.4f}")

    if mean_mse < best_mse:
        best_mse = mean_mse
        best_params = {
            'hidden_size': hidden_size,
            'context_size': context_size,
            'learning_rate': learning_rate,
            'alpha': alpha,
            'dropout': dropout,
            'weight_decay': weight_decay,
        }
        best_model_state = best_state_for_seed
        best_inv_predictions = inv_predictions
        best_inv_actuals = inv_actuals

# Print best result
print("\nBest Configuration:")
for k, v in best_params.items():
    print(f"{k}: {v}")
print(f"Best MSE: {best_mse:.4f}")

# Save best model
if best_model_state is not None:
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(best_model_state, "models/attention_gru_best.pth")

# Plot best model results
if not os.path.exists("plots"):
    os.makedirs("plots")

plot_predictions(best_inv_actuals, best_inv_predictions, title="Predicted vs Actual Prices (Best Model)", 
                 save_path="plots/best_predicted_vs_actual_prices.png")

print("Tuning complete!")