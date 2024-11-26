import torch
import torch.nn as nn
from .gru import GRU

class AttentionGRU(GRU):
    def __init__(self, input_size, hidden_size, context_size, output_size=None, device=None):
        # Adjust input_size to include context_size for concatenation
        super(AttentionGRU, self).__init__(input_size + context_size, hidden_size, output_size, device)
        self.context_size = context_size
        self.device = device

        self.Wa = nn.Parameter(torch.Tensor(hidden_size, context_size))
        self.Ua = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.va = nn.Parameter(torch.Tensor(hidden_size))

        self.initialize_attention()

    def initialize_attention(self):
        """ Initialize attention parameters """
        stdv = 1.0 / (self.hidden_size ** 0.5)
        nn.init.uniform_(self.Wa, -stdv, stdv)
        nn.init.uniform_(self.Ua, -stdv, stdv)
        nn.init.uniform_(self.va, -stdv, stdv)

    def attention(self, h: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.
        Arguments:
            h: hidden state at time t-1 (batch_size, hidden_size).
            context: context vectors to attend over (batch_size, context_seq_len, context_size).
        Returns:
            c_t: context vector at time t (batch_size, context_size).
            alpha_t: attention weights at time t (batch_size, context_seq_len).
        """
        batch_size, context_seq_len, _ = context.size()

        # Expand h to match context sequence length
        h_expanded = h.unsqueeze(1).expand(-1, context_seq_len, -1)  # (batch_size, context_seq_len, hidden_size)

        # Compute energy scores
        energy = torch.tanh(
            torch.matmul(context, self.Wa.t()) + torch.matmul(h_expanded, self.Ua.t())
        )  # (batch_size, context_seq_len, hidden_size)

        energies = torch.matmul(energy, self.va)  # (batch_size, context_seq_len)

        # Compute attention weights & context vector
        alpha_t = torch.softmax(energies, dim=1)  # (batch_size, context_seq_len)
        c_t = torch.bmm(alpha_t.unsqueeze(1), context).squeeze(1)  # (batch_size, context_size)

        return c_t, alpha_t

    def forward(self, x: torch.Tensor, context: torch.Tensor, h_prev: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a sequence with attention.
        Arguments:
            x: input sequence (batch_size, seq_len, input_size).
            context: context vectors to attend over (batch_size, context_seq_len, context_size).
            h_prev: initial hidden state (batch_size, hidden_size).
        Returns:
            outputs: output sequence (batch_size, seq_len, hidden_size or passed through output_layer).
            attention_weights: attention weights for each time step (batch_size, seq_len, context_seq_len).
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        h = torch.zeros(batch_size, self.hidden_size).to(device) if h_prev is None else h_prev

        outputs = []
        attention_weights = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)

            # Compute attention weights and context vector
            c_t, alpha_t = self.attention(h, context)
            attention_weights.append(alpha_t)

            x_combined = torch.cat((x_t, c_t), dim=1)  # (batch_size, input_size + context_size)

            # GRU cell computations
            z_t = torch.sigmoid(torch.matmul(x_combined, self.Wz) + torch.matmul(h, self.Uz) + self.bz)  # Update gate
            r_t = torch.sigmoid(torch.matmul(x_combined, self.Wr) + torch.matmul(h, self.Ur) + self.br)  # Reset gate
            h_cand = torch.tanh(torch.matmul(x_combined, self.Wh) + torch.matmul(r_t * h, self.Uh) + self.bh)  # Candidate hidden state
            h = (1 - z_t) * h + z_t * h_cand  # New hidden state

            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        attention_weights = torch.stack(attention_weights, dim=1)  # (batch_size, seq_len, context_seq_len)

        # Pass the outputs through the output layer if output_size is not None
        if self.output_layer:
            outputs = self.output_layer(outputs)

        return outputs, attention_weights