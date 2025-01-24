import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, device=None):
        """
        A custom implementation of a Gated Recurrent Unit (GRU) in PyTorch.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of features in the hidden state.
            output_size (int, optional): Number of output features (default: None).
            device (str, optional): Device to use for tensor computations (default: None).
        """
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # Define weights for update gate
        self.Wz = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Uz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bz = nn.Parameter(torch.Tensor(hidden_size))

        # Define weights for reset gate
        self.Wr = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Ur = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.br = nn.Parameter(torch.Tensor(hidden_size))

        # Define weights for candidate hidden state
        self.Wh = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Uh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bh = nn.Parameter(torch.Tensor(hidden_size))

        # Optional output layer
        self.output_layer = nn.Linear(hidden_size, output_size) if output_size else None

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize all weight parameters uniformly."""
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, h_prev=None):
        """Perform a forward pass for a sequence."""
        batch_size, seq_len, _ = x.size()
        h = h_prev if h_prev is not None else torch.zeros(batch_size, self.hidden_size, device=self.device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            z_t = torch.sigmoid(x_t @ self.Wz + h @ self.Uz + self.bz)  # Update gate
            r_t = torch.sigmoid(x_t @ self.Wr + h @ self.Ur + self.br)  # Reset gate
            h_candidate = torch.tanh(x_t @ self.Wh + (r_t * h) @ self.Uh + self.bh)  # Candidate hidden state
            h = (1 - z_t) * h + z_t * h_candidate  # Compute new hidden state

        return self.output_layer(h) if self.output_layer else h
