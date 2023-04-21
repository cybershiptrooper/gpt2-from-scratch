from model.attention import *
from model.embeddings import *
from model.layer_norm  import *
from model.transformer_mlp import *
from model.transformer_block import *
from model.transformer_model import *
from transformers import GPT2Model

# The methods in here can copy weights from huggingface's GPT2 model
# This was mainly made to run tests, but can be used for other stuff

class WeightCopier:
    def __init__(self, eval=True) -> None:
        self.reference_model = GPT2Model.from_pretrained('gpt2')
        if eval:
            self.reference_model.eval()
            self.reference_model.requires_grad_(False)

    def copy_word_embeddings(self, dst: WordEmbedding):
        src = self.reference_model.wte.weight
        dst.We = nn.Parameter(src.detach().clone())
        assert (dst.We == src).all()

    def copy_positional_embeddings(self, dst: PositionEmbedding):
        src = self.reference_model.wpe.weight
        dst.Pe = nn.Parameter(src.detach().clone())
        assert (dst.Pe == src).all()