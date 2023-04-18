from model.attention import *
from model.embeddings import *
from model.layer_norm  import *
from model.transformer_mlp import *
from model.transformer_block import *
from model.transformer_model import *
from utils.config import customConfig
import torch

d_model = 4
vocab_size = 10
batch_size = 2
words_per_sentence = 3

class TestShapes():
    def test_word_embeddings(self):
        embed = WordEmbedding(vocab_size, d_model)
        x = torch.randint(0, vocab_size, (batch_size, words_per_sentence)) 
        y = embed(x)
        # output shape: batch_size x words_per_sentence x d_model
        assert y.shape == (batch_size, words_per_sentence, d_model)

    def test_positional_embeddings(self):
        embed = PositionEmbedding(words_per_sentence, d_model)
        we_out = torch.rand(batch_size, words_per_sentence, d_model) 
        y = embed(we_out)
        # output shape: batch_size x words_per_sentence x d_model
        assert y.shape == (batch_size, words_per_sentence, d_model)

    def test_attention_head(self):
        attn = SingleHeadAttention(d_model, d_head=int(d_model/2), verbose=True)
        resid_strm = torch.rand(batch_size, words_per_sentence, d_model)
        y = attn(resid_strm)
        # output shape: batch_size x words_per_sentence x d_model
        assert y.shape == (batch_size, words_per_sentence, int(d_model/2))

    def test_multihead_attention(self):
        attn = MultiHeadAttention(d_model, n_heads=2, verbose=True)
        resid_strm = torch.rand(batch_size, words_per_sentence, d_model)
        y = attn(resid_strm)
        # output shape: batch_size x words_per_sentence x d_model
        assert y.shape == (batch_size, words_per_sentence, d_model)
    
    def test_layer_norm(self):
        ln = LayerNorm(d_model, verbose=True)
        x = torch.rand(batch_size, words_per_sentence, d_model)
        y = ln(x)
        # output shape: batch_size x words_per_sentence x d_model
        assert y.shape == (batch_size, words_per_sentence, d_model)

    def test_feedforward(self):
        mlp = FeedForward(d_model, d_mlp=2*d_model, verbose=True)
        x = torch.rand(batch_size, words_per_sentence, d_model)
        y = mlp(x)
        # output shape: batch_size x words_per_sentence x d_model
        assert y.shape == (batch_size, words_per_sentence, d_model)
    
    def test_transformer_block(self):
        cfg = customConfig({
            "d_model": d_model,
            "n_heads": 2,
            "d_head": int(d_model/2),
            "d_mlp": 2*d_model,
            "debug": True
        })
        assert cfg.d_model == d_model
        block = TransformerBlock(cfg)
        x = torch.rand(batch_size, words_per_sentence, d_model)
        y = block(x)
        # output shape: batch_size x words_per_sentence x d_model
        assert y.shape == (batch_size, words_per_sentence, d_model)

    def test_transformer_model(self):
        cfg = customConfig({
            "d_model": d_model,
            "n_heads": 2,
            "d_head": int(d_model/2),
            "d_mlp": 2*d_model,
            "debug": True,
            "n_ctx": words_per_sentence,
            "n_layers": 2,
            "vocab_size": vocab_size
        })
        assert cfg.d_model == d_model
        model = TransformerModel(cfg)
        x = torch.randint(0, vocab_size, (batch_size, words_per_sentence))
        y = model(x)
        # output shape: batch_size x words_per_sentence x vocab_size
        assert y.shape == (batch_size, words_per_sentence, vocab_size)


class TestValues(): # this probably is not the right way to test this...
    def test_word_embedding(self):
        embed = WordEmbedding(vocab_size, d_model)
        weight = embed.We
        x = torch.randint(0, vocab_size, (batch_size, words_per_sentence))  
        y = embed(x)
        assert torch.all(y == weight[x]) 

    def test_positional_embedding(self):
        embed = PositionEmbedding(words_per_sentence, d_model)
        Pe = embed.Pe
        we_out = torch.rand(batch_size, words_per_sentence, d_model) 
        y = embed(we_out)
        for b in range(batch_size):
            for i in range(words_per_sentence):
                # check if d_model dims of (y - we_out) is equal to Pe
                assert torch.isclose((y - we_out)[b, i, :], Pe[i], atol=1e-4, rtol=1e-3).all()