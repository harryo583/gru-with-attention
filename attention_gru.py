import torch 
import torch.nn as nn 
from gru import GRU

class AttentionGRU(GRU):
    def __init__(self, input_size, hidden_size, context_size):
        super(AttentionGRU, self).__init__(input_size + context_size, hidden_size)