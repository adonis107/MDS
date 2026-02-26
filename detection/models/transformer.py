import numpy as np
import torch
import torch.nn as nn

from detection.base import BaseDeepModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class BottleneckTransformerEncoder(nn.Module):
    """
    Transformer Encoder with Bottleneck Representation
    Takes (Batch, Sequence_length, Number_features) as input and outputs (Batch, Representation_dimension)
    """
    def __init__(self, num_features, model_dim, num_heads, num_layers, representation_dim, sequence_length):
        super(BottleneckTransformerEncoder, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(num_features, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.flatten = nn.Flatten()
        self.bottleneck = nn.Linear(sequence_length * model_dim, representation_dim)

    def forward(self, x):
        x = self.embedding(x) * np.sqrt(self.model_dim)

        x_pos = x.permute(1, 0, 2) # (Seq_len, Batch, Features)
        x_pos = self.pos_encoder(x_pos)
        x = x_pos.permute(1, 0, 2) # (Batch, Seq_len, Features)

        encoded_seq = self.transformer_encoder(x)

        flattened = self.flatten(encoded_seq)
        representation = self.bottleneck(flattened)

        return representation
    
    
class BottleneckTransformerDecoder(nn.Module):
    """
    Transformer Decoder with Bottleneck Representation
    Takes (Batch, Representation_dimension) as input and outputs (Batch, Sequence_length, Number_features)
    """
    def __init__(self, num_features, model_dim, num_heads, num_layers, representation_dim, sequence_length):
        super(BottleneckTransformerDecoder, self).__init__()
        self.model_dim = model_dim
        self.sequence_length = sequence_length

        self.expand = nn.Linear(representation_dim, sequence_length * model_dim)

        decoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layers, num_layers=num_layers)

        self.output_layer = nn.Linear(model_dim, num_features)

    def forward(self, x):
        expanded = self.expand(x)
        expanded = expanded.view(-1, self.sequence_length, self.model_dim)

        decoded_seq = self.transformer_decoder(expanded)

        output = self.output_layer(decoded_seq)

        return output
    

class BottleneckTransformer(BaseDeepModel):
    def __init__(self, num_features, model_dim, num_heads, num_layers, representation_dim, sequence_length):
        super(BottleneckTransformer, self).__init__()
        self.encoder = BottleneckTransformerEncoder(num_features, model_dim, num_heads, num_layers, representation_dim, sequence_length)
        self.decoder = BottleneckTransformerDecoder(num_features, model_dim, num_heads, num_layers, representation_dim, sequence_length)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        representation = self.encoder(x)
        reconstructed = self.decoder(representation)
        return reconstructed
    
    def get_representation(self, x):
        with torch.no_grad():
            return self.encoder(x)
        
    def training_step(self, batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(self.device)
        reconstruction = self.forward(x)
        loss = self.criterion(reconstruction, x)
        return loss
    
    def get_anomaly_score(self, batch):
        with torch.no_grad():
            x = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            reconstruction = self.forward(x)
            return torch.mean((reconstruction - x) ** 2, dim=(1, 2)).cpu().numpy()

