from dataclasses import dataclass, fields
import tiktoken as TT
from enum import Enum

class Activation(str, Enum):
    relu = "relu"
    gelu = "gelu"

@dataclass
class GPT2Config:
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_mlp: int = 3072 # 4 * d_model
    dropout: float = 0.1
    init_range: float = 0.02
    n_ctx: int = 1024
    layer_norm_eps: float = 1e-5
    debug: bool = True
    activation: Activation = Activation.gelu
    logits: bool = False

defaultConfig = GPT2Config()

def customConfig(argDict):
    className = GPT2Config
    fieldSet = {f.name for f in fields(className) if f.init}
    filteredArgDict = {k : v for k, v in argDict.items() if k in fieldSet}
    return className(**filteredArgDict)