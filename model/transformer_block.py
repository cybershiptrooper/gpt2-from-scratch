from .attention import MultiHeadAttention
from .layer_norm import LayerNorm
from .transformer_mlp import FeedForward
from utils.config import defaultConfig
from . import *

class TransformerBlock(nn.Module):
    def __init__(self, config=defaultConfig) -> None:
        super().__init__()
        self.config = config
        self.attn = MultiHeadAttention(config.d_model, config.n_heads, config.init_range, config.debug)
        self.ln1 = LayerNorm(config.d_model, config.layer_norm_eps, config.init_range, config.debug)
        self.mlp = FeedForward(config.d_model, config.d_mlp, config.init_range, config.debug)
        self.ln2 = LayerNorm(config.d_model, config.layer_norm_eps, config.init_range, config.debug)
        # self.dropout = nn.Dropout(config.dropout)
        self.net = torch.nn.Sequential(
            self.attn,
            self.ln1,
            self.mlp,
            self.ln2,
        )

    def forward(self, x):
        return self.net(x)