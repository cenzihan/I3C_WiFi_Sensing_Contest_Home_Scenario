import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CSITransformer(nn.Module):
    def __init__(self, input_features, d_model, n_heads, n_layers, dim_feedforward, num_rooms, num_classes_per_room, dropout=0.1):
        super(CSITransformer, self).__init__()
        self.d_model = d_model
        
        # Input embedding layer
        self.encoder = nn.Linear(input_features, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        
        # Output layers for each room
        self.classification_heads = nn.ModuleList([
            nn.Linear(d_model, num_classes_per_room) for _ in range(num_rooms)
        ])
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        for head in self.classification_heads:
            head.bias.data.zero_()
            head.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src shape: (batch_size, seq_len, features)
        src = self.encoder(src) * math.sqrt(self.d_model)
        
        # Add positional encoding. Note: PositionalEncoding module expects (seq_len, batch, dim)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        
        # Pass through the transformer encoder
        output = self.transformer_encoder(src)
        
        # We use the output corresponding to the first token ([CLS] token style) for classification
        output = output[0, :, :] # Take the first time step's output for all batches
        
        # Apply classification heads
        room_outputs = [head(output) for head in self.classification_heads]
        
        return room_outputs