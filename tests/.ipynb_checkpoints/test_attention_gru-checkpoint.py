import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models.attention_gru import AttentionGRU
from losses.correlation_loss import CorrelationLoss
from losses.hybrid_loss import HybridLoss

# Read data
current_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(current_dir, "../datasets/DailyDelhiClimateTrain.csv")
test_path = os.path.join(current_dir, "../datasets/DailyDelhiClimateTest.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Preprocessing
def normalize(df, feature_means=None, feature_stds=None):
    if feature_means is None or feature_stds is None:
        feature_means = df.mean()
        feature_stds = df.std()
    return (df - feature_means) / feature_stds, feature_means, feature_stds

train_features = train[['meantemp', 'humidity', 'wind_speed', 'meanpressure']]
train_targets = train[['meantemp', 'humidity']]
train_features, train_means, train_stds = normalize(train_features)
train_targets, _, _ = normalize(train_targets, train_means[['meantemp', 'humidity']], train_stds[['meantemp', 'humidity']])

test_features = test[['meantemp', 'humidity', 'wind_speed', 'meanpressure']]
test_targets = test[['meantemp', 'humidity']]
test_features, _, _ = normalize(test_features, train_means, train_stds)
test_targets, _, _ = normalize(test_targets, train_means[['meantemp', 'humidity']], train_stds[['meantemp', 'humidity']])

train_features = torch.tensor(train_features.values, dtype=torch.float32)
train_targets = torch.tensor(train_targets.values, dtype=torch.float32)
test_features = torch.tensor(test_features.values, dtype=torch.float32)
test_targets = torch.tensor(test_targets.values, dtype=torch.float32)

# Create sequences for AttentionGRU
def create_sequences(features, targets, seq_length=5):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(targets[i + seq_length])
    return torch.stack(X), torch.stack(y)

seq_length = 5
X_train, y_train = create_sequences(train_features, train_targets, seq_length)
X_test, y_test = create_sequences(test_features, test_targets, seq_length)

device = torch.device("cpu")
input_size = 4
context_size = input_size  # assuming context comes from the same input features
hidden_size = 16
output_size = 2  # meantemp and humidity

# Initialize the AttentionGRU model & define loss and optimizer
attention_gru_model = AttentionGRU(input_size, hidden_size, context_size, output_size, device).to(device)
criterion = HybridLoss(0.3)  # Adjust weight if necessary
optimizer = optim.Adam(attention_gru_model.parameters(), lr=0.001)

# Training
num_epochs = 100
batch_size = 16

for epoch in range(num_epochs):
    attention_gru_model.train()
    epoch_loss = 0
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size].to(device)  # (batch_size, seq_length, input_size)
        y_batch = y_train[i:i + batch_size].to(device)  # (batch_size, output_size)
        
        context = X_batch  # (batch_size, seq_length, input_size)
        
        optimizer.zero_grad()
        outputs, _ = attention_gru_model(X_batch, context)  # outputs: (batch_size, seq_length, output_size)
        
        # NOTE We might need to adjust outputs and y_batch dimensions since outputs are per timestep, but y_batch is per sequence
        outputs = outputs[:, -1, :]  # (batch_size, output_size)
        
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(attention_gru_model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(X_train)}")

# Testing
attention_gru_model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    context_test = X_test
    predictions, attention_weights = attention_gru_model(X_test, context_test)
    predictions = predictions[:, -1, :]
    test_loss = criterion(predictions, y_test)
    print(f"Test Loss: {test_loss.item()}")

# Plot results
def plot_results(y_true, y_pred, feature_name, feature_index):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[:, feature_index], label="Actual", linestyle='-', marker='o')
    plt.plot(y_pred[:, feature_index], label="Predicted", linestyle='--', marker='x')
    plt.title(f"Actual vs Predicted {feature_name}", fontsize=16)
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel(feature_name, fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()

plot_results(y_test, predictions, "Mean Temperature", 0)
plot_results(y_test, predictions, "Humidity", 1)
