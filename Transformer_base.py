# a pytorch transformer base class
import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout
        )
        self.decoder = nn.TransformerDecoder(
            nhead,
            num_decoder_layers,
            dim_feedforward,
            dropout
        )
        self.encoder.final_norm = False
        self.decoder.final_norm = False
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.dropout = dropout

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if src_mask is None:
            src_mask = torch.ones(src.size(0), src.size(1), dtype=torch.uint8, device=src.device)
        if tgt_mask is None:
            tgt_mask = torch.ones(tgt.size(0), tgt.size(1), dtype=torch.uint8, device=tgt.device)
        if memory_mask is None:
            memory_mask = torch.ones(src.size(0), src.size(1), dtype=torch.uint8, device=src.device)
        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(src.size(0), src.size(1), dtype=torch
                                                  .uint8, device=src.device)
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = torch.zeros(tgt.size(0), tgt.size(1), dtype=torch
                                                  .uint8, device=tgt.device)
        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.zeros(src.size(0), src.size(1), dtype=torch
                                                  .uint8, device=src.device)
        src = self.encoder(src, src_mask, src_key_padding_mask)
        tgt = self.decoder(tgt, memory=src, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return tgt

def train_transformer(num_epochs=300):
    model = Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        for batch in range(10):
            src = torch.rand(64, 64, dtype=torch.long, device='cuda')
            tgt = torch.rand(64, 64, dtype=torch.long, device='cuda')
            output = model(src, tgt)
            loss = loss_fn(output, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, batch, loss)
