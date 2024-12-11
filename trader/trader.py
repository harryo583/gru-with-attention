import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from models.attention_gru import AttentionGRU

# random.seed(14)

##############################
# Parameters and Data Loading
##############################
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
    Y.append(scaled_features[i+seq_len, 3])  # Next day's close

X = np.array(X)
Y = np.array(Y)

train_size = int(0.8 * len(X))
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

#############
# Load Model
#############
hidden_size = 64
context_size = 5
model = AttentionGRU(input_size=input_size, hidden_size=hidden_size,
                     context_size=context_size, output_size=output_size, device=device).to(device)
model.load_state_dict(torch.load("models/attention_gru_advanced.pth", map_location=device))
model.eval()

#######################
# Generate Predictions
#######################
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)

predictions = []
with torch.no_grad():
    # Process in batches to avoid memory issues
    for start_idx in range(0, len(X_test_tensor), batch_size):
        end_idx = start_idx + batch_size
        x_batch = X_test_tensor[start_idx:end_idx]
        context_batch = x_batch.clone()  # context is often same as input
        outputs, attn_weights = model(x_batch, context_batch)
        pred = outputs[:, -1, 0]
        predictions.append(pred.cpu().numpy())

predictions = np.concatenate(predictions)
actuals = Y_test  # already in CPU numpy

# Inverse transform predictions and actuals
dummy = np.zeros((len(predictions), input_size))
dummy[:, 3] = predictions
inv_predictions = scaler.inverse_transform(dummy)[:, 3]

dummy[:, 3] = actuals
inv_actuals = scaler.inverse_transform(dummy)[:, 3]

# Get corresponding open prices for these predicted days
start_idx_global = train_size + seq_len
test_opens = df['Open'].values[start_idx_global:start_idx_global+len(inv_predictions)]
test_closes = df['Close'].values[start_idx_global:start_idx_global+len(inv_predictions)]

#########################
# Naive Trading Strategy
#########################
initial_capital = 100.0
current_capital = initial_capital
window_size = 200

if len(inv_predictions) > window_size:
    start_day = random.randint(0, 100)
    start_day = 0
else:
    start_day = 0
end_day = start_day + window_size

capital_over_time = []
for i in range(start_day, end_day):
    predicted_close = inv_predictions[i]
    actual_open = test_opens[i]
    actual_close = test_closes[i]

    if predicted_close > actual_open:
        shares_bought = current_capital / actual_open
        end_capital = shares_bought * actual_close
        current_capital = end_capital

    capital_over_time.append(current_capital)

############################
# Plot the Trader's Returns
############################
plt.figure(figsize=(10, 6))
plt.plot(range(1, window_size+1), capital_over_time, marker='o', label='Trader Capital')
plt.title(f'Returns over {window_size}-day window')
plt.xlabel('Day')
plt.ylabel('Capital')
plt.grid(True)
plt.legend()
plt.show()
