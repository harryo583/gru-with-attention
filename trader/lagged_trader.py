import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler

##############################
# Parameters and Data Loading
##############################
csv_file = "datasets/BTC-USD.csv"
seq_len = 30
batch_size = 32

# Load and preprocess the dataset
df = pd.read_csv(csv_file)
df = df.sort_values('Date')

# Scale the features
features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# Prepare test data (no training required for lagged strategy)
train_size = int(0.8 * len(scaled_features))
start_idx_global = train_size + seq_len

test_opens = df['Open'].values[start_idx_global:]
test_closes = df['Close'].values[start_idx_global:]

#########################
# Lagged Trading Strategy
#########################
initial_capital = 100.0
current_capital = initial_capital
window_size = 200

if len(test_opens) > window_size:
    start_day = random.randint(0, len(test_opens) - window_size)
else:
    start_day = 0
end_day = start_day + window_size

capital_over_time = []
for i in range(start_day + 1, end_day):
    yesterday_close = test_closes[i - 1]
    today_open = test_opens[i]
    today_close = test_closes[i]

    if today_open < yesterday_close:
        shares_bought = current_capital / today_open
        end_capital = shares_bought * today_close
        current_capital = end_capital

    capital_over_time.append(current_capital)

###############
# Plot Returns
###############
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(capital_over_time) + 1), capital_over_time, marker='o', label='Trader Capital')
plt.title(f'Returns over {window_size}-day window')
plt.xlabel('Day')
plt.ylabel('Capital')
plt.grid(True)
plt.legend()
plt.show()
