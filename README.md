# GRU with Attention  
**Northwestern CS_449 Final Project**

This repository implements a GRU-based time-series prediction model enhanced with attention mechanisms. The project focuses on leveraging machine learning techniques to model and forecast trends in time series data, particularly for trading.  

---

## Directory Structure  

```
.
├── bitcoin/            # Code and scripts related to processing Bitcoin data
├── datasets/           # Data files or data loading scripts
├── losses/             # Custom loss functions used for training the model
│   ├── correlation_loss.py
│   ├── hybrid_loss.py
├── models/             # Implementation of GRU and Attention GRU models
│   ├── __init__.py
│   ├── attention_gru.py
│   ├── gru.py
├── plots/              # Scripts or generated plots for visualizing results
├── saved_models/       # Saved weights or checkpoints of trained models
├── tests/              # Unit tests for verifying model components
├── trader/             # Trading logic and evaluation scripts
└── .gitignore          # Git configuration file for ignoring unnecessary files
```

---

## Features  

- **Models**:  
  - `gru.py`: A custom implementation of the Gated Recurrent Unit (GRU) architecture for time-series prediction.  
  - `attention_gru.py`: An extension of the GRU model, incorporating an attention mechanism to enhance the performance on sequential data.  

- **Loss Functions**:  
  - `correlation_loss.py`: A custom loss function designed to optimize correlation between predicted and target values.  
  - `hybrid_loss.py`: A combined loss function that blends multiple objectives for robust training.  

- **Bitcoin Analysis**:  
  Scripts in the `bitcoin` directory provide tools for processing, analyzing, and predicting Bitcoin trends.

- **Trading Logic**:  
  The `trader` directory contains scripts that evaluate the models in a simulated trading environment.

- **Visualization**:  
  The `plots` directory contains tools for visualizing model performance and predictions.  

- **Saved Models**:  
  Trained model checkpoints are stored in `saved_models` for reproducibility and future use.

---

## Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/harryo583/gru-with-attention
   cd gru-with-attention
   ```

2. Install required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Set up datasets by placing data files in the `datasets/` directory.

---

## Usage  

### Training  

1. Configure the model and training parameters in the appropriate scripts under `models/`.  
2. Train the model:  
   ```bash
   python train.py
   ```

### Evaluation  

Evaluate the model's performance using the test scripts under the `tests/` or `trader/` directories.

### Visualization  

Generate and view plots using scripts in the `plots/` directory.  

---

## Acknowledgments  

This project was developed as part of the **Northwestern CS_449 Final Project**. Special thanks to the course instructors and teaching assistants for their guidance.

---
