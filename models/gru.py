import torch 
import torch.nn as nn 

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, device=None):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        
        # Weights for the update gate
        self.Wz = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Uz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bz = nn.Parameter(torch.Tensor(hidden_size))
        
        # Weights for the reset gate
        self.Wr = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Ur = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.br = nn.Parameter(torch.Tensor(hidden_size))
        
        # Weights for the candidate hidden state
        self.Wh = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Uh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bh = nn.Parameter(torch.Tensor(hidden_size))
        
        # Optional fully connected output layer
        self.output_layer = None
        if output_size is not None:
            self.output_layer = nn.Linear(hidden_size, output_size)
        
        self.initialize()
    
    def initialize(self):
        """ Initialize the weight parameters """
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
    
    def forward(self, x, h_prev=None):
        """ Forward pass for a sequence """
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size).to(self.device) if h_prev is None else h_prev
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            z_t = torch.sigmoid(torch.matmul(x_t, self.Wz) + torch.matmul(h, self.Uz) + self.bz)  # update gate
            r_t = torch.sigmoid(torch.matmul(x_t, self.Wr) + torch.matmul(h, self.Ur) + self.br)  # reset gate
            h_cand = torch.tanh(torch.matmul(x_t, self.Wh) + torch.matmul(r_t * h, self.Uh) + self.bh)  # candidate hidden unit
            h = (1 - z_t) * h + z_t * h_cand  # compute new hidden state

        # Optionally passes the final hidden state through the output layer
        if self.output_layer:
            return self.output_layer(h)
        return h
