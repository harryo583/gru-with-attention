# NOTE RUN python -m bitcoin.attention_gru_simple in the ROOT DIRECTORY

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from models.attention_gru import AttentionGRU
from losses.hybrid_loss import HybridLoss
from losses.correlation_loss import CorrelationLoss

#############
# Parameters
#############
csv_file = "datasets/BTC-USD.csv"
seq_len = 30
batch_size = 32
hidden_size = 64
context_size = 5
input_size = 5
output_size = 1
num_epochs = 30
learning_rate = 0.001
alpha = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################
# Data Loading and Preprocessing
#################################

df = pd.read_csv(csv_file)
df = df.sort_values('Date')

features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

X, Y = [], []
for i in range(len(scaled_features) - seq_len):
    X.append(scaled_features[i:i+seq_len, :])   
    Y.append(scaled_features[i+seq_len, 3])

X = np.array(X)
Y = np.array(Y)

train_size = int(0.8 * len(X))
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

############################
# Initialize Model and Loss
############################
model = AttentionGRU(input_size=input_size, hidden_size=hidden_size, 
                     context_size=context_size, output_size=output_size, device=device).to(device)

criterion = HybridLoss(alpha=alpha)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

################
# Training Loop
################

train_losses = []

for epoch in range(num_epochs):
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
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

#############
# Evaluation
#############

model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
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
print(f"Test MSE: {mse:.4f}")

#############
# Save Model
#############
torch.save(model.state_dict(), "saved_models/attention_gru.pth")

#####################
# Plot Training Loss
#####################
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig("training_loss_plot.png")
plt.show()


##########################################
# Backtesting: Predicted vs Actual Prices
##########################################
plt.figure(figsize=(14, 8))
plt.plot(inv_actuals, label="Actual Prices", color="blue", alpha=0.7)
plt.plot(inv_predictions, label="Predicted Prices", color="orange", alpha=0.7)
plt.title("Predicted vs Actual Prices (Backtesting)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.savefig("plots/predicted_vs_actual_prices.png")
plt.show()