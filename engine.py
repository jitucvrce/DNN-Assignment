import torch
import torch.nn as nn
import numpy as np
import time

# =====================================================================
# TASK 1: SYNTHETIC GENERATORS
# =====================================================================

def generate_adding_problem(num_sequences, length):
    """Generates synthetic distributions for the sequence addition verification task."""
    X_val = np.random.uniform(0, 1, (num_sequences, length, 1))
    X_mark = np.zeros((num_sequences, length, 1))
    Y = np.zeros((num_sequences, 1))
    
    half = length // 2
    for i in range(num_sequences):
        idx1 = np.random.randint(0, half)
        idx2 = np.random.randint(half, length)
        X_mark[i, idx1, 0] = 1.0
        X_mark[i, idx2, 0] = 1.0
        Y[i, 0] = X_val[i, idx1, 0] + X_val[i, idx2, 0]
        
    X = np.concatenate([X_val, X_mark], axis=-1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def generate_mock_har(num_samples=600, sequence_length=128, input_dim=9):
    """Simulates real-world normalized Human Activity Recognition telemetry."""
    X = np.random.randn(num_samples, sequence_length, input_dim)
    y = np.random.randint(0, 6, size=(num_samples,))
    subjects = np.random.randint(1, 31, size=(num_samples,))
    return X, y, subjects

# =====================================================================
# TASK 2: CUSTOM RECURRENT CELL ALGEBRAIC DEFINITIONS
# =====================================================================

class MyRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.weight_ih = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.empty(hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(hidden_dim))

    def forward(self, x, h):
        return torch.tanh(torch.matmul(x, self.weight_ih.t()) + self.bias_ih + 
                          torch.matmul(h, self.weight_hh.t()) + self.bias_hh)

class MyLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_dim))

    def forward(self, x, h_c):
        h, c = h_c
        gates = (torch.matmul(x, self.weight_ih.t()) + self.bias_ih + 
                 torch.matmul(h, self.weight_hh.t()) + self.bias_hh)
        i, f, g, o = gates.chunk(4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

class MyGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_dim))

    def forward(self, x, h):
        w_ih_r, w_ih_z, w_ih_n = self.weight_ih.chunk(3, dim=0)
        w_hh_r, w_hh_z, w_hh_n = self.weight_hh.chunk(3, dim=0)
        b_ih_r, b_ih_z, b_ih_n = self.bias_ih.chunk(3, dim=0)
        b_hh_r, b_hh_z, b_hh_n = self.bias_hh.chunk(3, dim=0)
        
        r = torch.sigmoid(torch.matmul(x, w_ih_r.t()) + b_ih_r + torch.matmul(h, w_hh_r.t()) + b_hh_r)
        z = torch.sigmoid(torch.matmul(x, w_ih_z.t()) + b_ih_z + torch.matmul(h, w_hh_z.t()) + b_hh_z)
        n = torch.tanh(torch.matmul(x, w_ih_n.t()) + b_ih_n + r * (torch.matmul(h, w_hh_n.t()) + b_hh_n))
        return (1 - z) * n + z * h

# =====================================================================
# SYSTEM MODEL WRAPPERS
# =====================================================================

class UnifiedSequenceModel(nn.Module):
    def __init__(self, cell_type, input_dim, hidden_dim, output_dim, forget_bias_init=1.0):
        super().__init__()
        self.cell_type = cell_type
        self.hidden_dim = hidden_dim
        
        if cell_type == "RNN":
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        elif cell_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            with torch.no_grad():
                self.rnn.bias_ih_l0[hidden_dim:2*hidden_dim].fill_(forget_bias_init)
                self.rnn.bias_hh_l0[hidden_dim:2*hidden_dim].fill_(forget_bias_init)
        elif cell_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
            
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
