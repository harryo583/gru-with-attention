# naive trader who buys when predicted close price is higher than opening price and profits off the difference

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from models.attention_gru import AttentionGRU

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(14)

#################################
# Parameters and Data Loading
#################################

csv_file = "datasets/BTC-USD.csv"
seq_len = 30
input_size = 5
output_size = 1
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(csv_file)
df = df.sort_values('Date')

features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

X, Y = [], []
for i in range(len(scaled_features) - seq_len):
    X.append(scaled_features[i:i+seq_len, :])
    Y.append(scaled_features[i+seq_len, 3])  # Closing price

X = np.array(X)
Y = np.array(Y)

train_size = int(0.8 * len(X))
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

class BitcoinDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        context = torch.tensor(self.X[idx], dtype=torch.float32)
        return x, context, y

test_dataset = BitcoinDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#############
# Load Model
#############

# Hyperparameters should match what you used during training
hidden_size = 64
context_size = 5
model = AttentionGRU(input_size=input_size, hidden_size=hidden_size,
                     context_size=context_size, output_size=output_size, device=device).to(device)

model.load_state_dict(torch.load("models/attention_gru_advanced.pth", map_location=device))
model.eval()

###################################
# Generate Predictions on Test Set
###################################

predictions, actuals = [], []
with torch.no_grad():
    for x_batch, context_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        context_batch = context_batch.to(device)
        y_batch = y_batch.to(device)

        outputs, attn_weights = model(x_batch, context_batch)
        pred = outputs[:, -1, 0]  # Last day's prediction in the sequence
        predictions.append(pred.cpu().numpy())
        actuals.append(y_batch.cpu().numpy())

predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)

# Inverse transform predictions and actuals
dummy = np.zeros((len(predictions), input_size))
dummy[:,3] = predictions
inv_predictions = scaler.inverse_transform(dummy)[:,3]

dummy[:,3] = actuals
inv_actuals = scaler.inverse_transform(dummy)[:,3]

# Get corresponding open prices for these predicted days
# The predictions correspond to days starting from `train_size + seq_len`
start_idx = train_size + seq_len
test_opens = df['Open'].values[start_idx:start_idx+len(inv_predictions)]
test_closes = df['Close'].values[start_idx:start_idx+len(inv_predictions)]

#############################
# Implement the Naive Trader
#############################

initial_capital = 100.0
capital = initial_capital
window_size = 200

# Choose a random 100-day window in the test predictions
if len(inv_predictions) > window_size:
    # start_day = np.random.randint(0, len(inv_predictions)-window_size)
    start_day = 1
else:
    start_day = 0
end_day = start_day + window_size

capital_over_time = []
current_capital = capital

for i in range(start_day, end_day):
    # If predicted close price > today's actual opening price -> buy at open and sell at close
    predicted_close = inv_predictions[i]
    actual_open = test_opens[i]
    actual_close = test_closes[i]

    if predicted_close > actual_open:
        # Buy as much as we can at open
        shares_bought = current_capital / actual_open
        # Sell all at close
        end_capital = shares_bought * actual_close
        current_capital = end_capital

    capital_over_time.append(current_capital)

#################################
# Plot the Trader's Returns
#################################

plt.figure(figsize=(10, 6))
plt.plot(range(1, window_size+1), capital_over_time, marker='o', label='Trader Capital')
plt.title(f'Naive Trader Returns Over {window_size}-Day Window')
plt.xlabel('Day')
plt.ylabel('Capital')
plt.grid(True)
plt.legend()
plt.show()
