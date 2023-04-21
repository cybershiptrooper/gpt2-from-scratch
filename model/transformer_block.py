from .attention import MultiHeadAttention
from .layer_norm import LayerNorm
from .transformer_mlp import FeedForward
from utils.config import defaultConfig
from . import *

class TransformerBlock(nn.Module):
    def __init__(self, config=defaultConfig) -> None:
        super().__init__()
        self.config = config
        self.ln1 = LayerNorm(config.d_model, config.layer_norm_eps, config.debug)
        self.attn = MultiHeadAttention(config.d_model, config.n_heads, config.init_range, config.debug)
        self.ln2 = LayerNorm(config.d_model, config.layer_norm_eps, config.debug)
        self.mlp = FeedForward(config.d_model, config.d_mlp, config.init_range, config.debug)
        # self.dropout = nn.Dropout(config.dropout)
        self.net = torch.nn.Sequential(
            self.ln1,
            self.attn,
            self.ln2,
            self.mlp,
        )

    def forward(self, x):
        return self.net(x)