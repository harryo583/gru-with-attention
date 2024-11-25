import os
import pandas as pd
from models.gru import GRU

### Read Data

current_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(current_dir, "../datasets/DailyDelhiClimateTrain.csv")
test_path = os.path.join(current_dir, "../datasets/DailyDelhiClimateTest.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(train.head())
print(test.head())