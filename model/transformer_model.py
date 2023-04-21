from . import *
from .transformer_block import TransformerBlock
from .embeddings import WordEmbedding, PositionEmbedding
from .layer_norm import LayerNorm
from utils.config import defaultConfig

class TransformerModel(nn.Module):
    def __init__(self, config = defaultConfig):
        super().__init__()
        self.config = config
        self.we = WordEmbedding(config.vocab_size, config.d_model, config.init_range, config.debug)
        self.pe = PositionEmbedding(config.n_ctx, config.d_model, config.init_range, config.debug)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.lnf = LayerNorm(config.d_model, config.layer_norm_eps, config.debug)
        self.unembed = self.we.We
        self.net = torch.nn.Sequential(
            self.we,
            self.pe,
            *self.blocks, 
            self.lnf
        )
    def forward(self, x):
        x = self.net(x)
        x = x @ self.unembed.T
        if(self.config.logits):
            return x
        else:
            return nn.Softmax(dim=-1)(x)