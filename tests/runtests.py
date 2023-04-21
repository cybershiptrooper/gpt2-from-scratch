from model.attention import *
from model.embeddings import *
from model.layer_norm  import *
from model.transformer_mlp import *
from model.transformer_block import *
from model.transformer_model import *
from utils.config import customConfig
import torch
from transformers import GPT2Tokenizer
from utils.copy_weights import *

d_model = 4
vocab_size = 10
batch_size = 2
words_per_sentence = 3
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu'
torch.manual_seed(0)
class TestShapes():
    def test_word_embeddings(self):
        embed = WordEmbedding(vocab_size, d_model)
        embed.to(device)
        x = torch.randint(0, vocab_size, (batch_size, words_per_sentence)).to(device) 
        y = embed(x)
        # output shape: batch_size x words_per_sentence x d_model
        assert y.shape == (batch_size, words_per_sentence, d_model)

    def test_positional_embeddings(self):
        embed = PositionEmbedding(words_per_sentence, d_model)
        embed.to(device)
        we_out = torch.rand(batch_size, words_per_sentence, d_model).to(device) 
        y = embed(we_out)
        # output shape: batch_size x words_per_sentence x d_model
        assert y.shape == (batch_size, words_per_sentence, d_model)

    def test_attention_head(self):
        attn = SingleHeadAttention(d_model, d_head=int(d_model/2), verbose=True)
        attn.to(device)
        resid_strm = torch.rand(batch_size, words_per_sentence, d_model).to(device)
        y = attn(resid_strm)
        # output shape: batch_size x words_per_sentence x d_model
        assert y.shape == (batch_size, words_per_sentence, int(d_model/2))

    def test_multihead_attention(self):
        attn = MultiHeadAttention(d_model, n_heads=2, verbose=True)
        attn.to(device)
        resid_strm = torch.rand(batch_size, words_per_sentence, d_model).to(device)
        y = attn(resid_strm)
        # output shape: batch_size x words_per_sentence x d_model
        assert y.shape == (batch_size, words_per_sentence, d_model)
    
    def test_layer_norm(self):
        ln = LayerNorm(d_model, verbose=True)
        ln.to(device)
        x = torch.rand(batch_size, words_per_sentence, d_model).to(device)
        y = ln(x)
        # output shape: batch_size x words_per_sentence x d_model
        assert y.shape == (batch_size, words_per_sentence, d_model)

    def test_feedforward(self):
        mlp = FeedForward(d_model, d_mlp=2*d_model, verbose=True)
        mlp.to(device)
        x = torch.rand(batch_size, words_per_sentence, d_model).to(device)
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
        block.to(device)
        x = torch.rand(batch_size, words_per_sentence, d_model).to(device)
        y = block(x)
        # output shape: batch_size x words_per_sentence x d_model
        assert y.shape == (batch_size, words_per_sentence, d_model)

    def test_transformer_model(self):
        cfg = customConfig({
            # "d_model": d_model,
            # "n_heads": 2,
            # "d_head": int(d_model/2),
            # "d_mlp": 2*d_model,
            "debug": True,
            "init_range": 0.02/((2 * defaultConfig.n_layers)**0.5),
            # "n_ctx": words_per_sentence,
            # "n_layers": 2,
            # "vocab_size": vocab_size
        })
        # assert cfg.d_model == d_model
        model = TransformerModel(cfg)
        model.to(device)
        x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.n_ctx)).to(device)
        y = model(x)
        # output shape: batch_size x words_per_sentence x vocab_size
        assert y.shape == (batch_size, cfg.n_ctx, cfg.vocab_size)


class TestValues(): # this probably is not the right way to test this...
    def test_word_embedding(self):
        embed = WordEmbedding(vocab_size, d_model)
        embed.to(device)
        weight = embed.We
        x = torch.randint(0, vocab_size, (batch_size, words_per_sentence)).to(device)  
        y = embed(x)
        assert torch.all(y == weight[x]) 

    def test_positional_embedding(self): 
        embed = PositionEmbedding(words_per_sentence, d_model)
        embed.to(device)
        Pe = embed.Pe
        we_out = torch.rand(batch_size, words_per_sentence, d_model).to(device) 
        y = embed(we_out)
        for b in range(batch_size):
            for i in range(words_per_sentence):
                # check if d_model dims of (y - we_out) is equal to Pe
                assert torch.isclose((y - we_out)[b, i, :], Pe[i], atol=1e-4, rtol=1e-3).all()

# This is vv adhoc
class TestValuesUsingReference():
    def setup_reference(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
        self.input = "This is a test sentence. This is"
        self.input_tokens = self.tokenizer(self.input, return_tensors='pt')['input_ids']
        self.copier = WeightCopier()
        self.reference_model = self.copier.reference_model

    def test_word_embedding(self):
        self.setup_reference()
        ref = self.reference_model.wte(self.input_tokens)
        embed = WordEmbedding(defaultConfig.vocab_size, defaultConfig.d_model)
        # embed.to(device) # ideally, we should check across devices, but my architecture sucks rn
        self.copier.copy_word_embeddings(embed)
        with torch.no_grad():
            out = embed(self.input_tokens)
        assert (out == ref).all()

    def test_positional_embedding(self):  
        self.setup_reference()
        we_out = torch.zeros(2, defaultConfig.n_ctx, defaultConfig.d_model, dtype=int) # b x words_per_sentence x d_model
        # Hugging faces takes range as input directly:
        ref_input = torch.stack([torch.arange(we_out.shape[1]) for _ in range(we_out.shape[0])]) 
        ref = self.reference_model.wpe(ref_input)
        embed = PositionEmbedding(defaultConfig.n_ctx, defaultConfig.d_model)
        # embed.to(device) # ideally, we should check across devices, but my architecture sucks rn
        self.copier.copy_positional_embeddings(embed)
        with torch.no_grad():
            out = embed(we_out)
        assert (out == ref).all()
    
    def test_layer_norm(self):
        layer = LayerNorm(d_model, verbose=False).to(device)
        ref = nn.LayerNorm(d_model, eps=1e-12).to(device)
        assert(layer.gamma == ref.weight).all()
        assert(layer.beta == ref.bias).all() # both init to ones and zeros vectors
        x = torch.rand(batch_size, words_per_sentence, d_model).to(device)
        with torch.no_grad():
            out = layer(x)
            ref_out = ref(x)
        assert torch.isclose(out, ref_out, atol=1e-4, rtol=1e-3).all()

    def test_multihead_attention(self):
        # self.setup_reference()
        # ref = self.reference_model.h[0].attn(self.input_tokens)
        # attn = MultiHeadAttention(defaultConfig.d_model, n_heads=defaultConfig.n_heads)
        # # attn.to(device) # ideally, we should check across devices, but my architecture sucks rn
        # self.copier.copy_multihead_attention(attn)
        # with torch.no_grad():
        #     out = attn(self.input_tokens)
        # assert (out == ref).all()
        pass