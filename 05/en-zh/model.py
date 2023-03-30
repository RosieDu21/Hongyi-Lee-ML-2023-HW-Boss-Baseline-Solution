import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Type
from config import Config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self, config: Type[Config]) -> None:
        super(Model, self).__init__()
        self.d_model = config.hidden_width
        self.n_head  = config.n_head

        self.embed = nn.Embedding(config.vocab_size, config.hidden_width)

        self.pe = PositionalEncoding(config.hidden_width, config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(config.hidden_width,config.n_head,config.feedforward,config.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.hidden_depth,norm=nn.LayerNorm(config.hidden_width))

        decoder_layer = nn.TransformerDecoderLayer(config.hidden_width,config.n_head,config.feedforward,config.dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.hidden_depth,norm=nn.LayerNorm(config.hidden_width))

        self.output = nn.Linear(config.hidden_width,config.vocab_size)

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self,
            x: torch.Tensor,
            y: torch.Tensor,
            src_padding_mask: torch.Tensor = None,
            tgt_padding_mask: torch.Tensor = None,
            require_memory: bool = False,
            is_x_memory: bool = False
        ) -> Union[tuple[torch.Tensor,torch.Tensor], torch.Tensor]:
        if not is_x_memory:
            # (len, batch)
            x = x.T
            # (len, batch, d_model)
            x = self.embed(x) * math.sqrt(self.d_model)
            x = self.pe(x)
            # (len, batch, d_model)
            mem = self.encoder(x,src_key_padding_mask=src_padding_mask)
        else:
            # (len, batch, d_model)
            mem = x.permute(1,0,2)

        # (len, batch)
        y = y.T
        # (len, batch, d_model)
        y = self.embed(y) * math.sqrt(self.d_model)
        y = self.pe(y)

        # (len, batch, d_model)
        attn_mask = torch.triu(torch.ones((y.size(0),y.size(0)),dtype=torch.bool).to(y.device), diagonal=1)
        out = self.decoder(
            y,
            mem,
            tgt_mask=attn_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )

        # out = out / math.sqrt(self.d_model / self.n_head)

        # (batch, len, d_model)
        mem = mem.permute(1,0,2)
        out = out.permute(1,0,2)
        # (batch, len, vocab)
        # out = torch.einsum('vd,bld->blv',(F.normalize(self.embed.weight.data, dim=-1), F.normalize(out, dim=-1)))
        out = self.output(out)

        if require_memory:
            return out, mem
        else:
            return out