from . import *

class WordEmbedding(nn.Module):
    '''
    Word embedding layer for gpt2
    '''
    def __init__(self, vocab_size, d_model, init_range=0.02, verbose = False):
        super().__init__()
        self.We = nn.Parameter(torch.empty(vocab_size, d_model))
        self.verbose = verbose
        nn.init.normal_(self.We, mean=0, std=init_range)
        
    def forward(self, tokens):
        # tokens shape: batch_size x words_per_sentence
        if(self.verbose):
            print("Embedding layer input shape: ", tokens.shape)

        # Embedding layer picks the nth row of the embedding matrix for each token in the sentence
        # Output shape: batch_size x words_per_sentence x d_model 
        # So the token(last dim) gets inflated to a vector of size d_model
        out = self.We[tokens]
        if(self.verbose):
            print("Embedding layer output shape: ", out.shape)
        return out

class PositionEmbedding(nn.Module):
    '''
    Positional embedding layer for gpt2
    gpt uses learned positional embeddings
    ''' 
    def __init__(self, n_ctx, d_model, init_range=0.02, verbose = False):
        super().__init__()
        self.Pe = nn.Parameter(torch.empty(n_ctx, d_model))
        self.verbose = verbose
        nn.init.normal_(self.Pe, mean=0, std=init_range)

    def forward(self, word_embeddings):
        # word_embeddings shape: batch_size x words_per_sentence x d_model
        if(self.verbose):
            print("Positional embedding layer input shape: ", word_embeddings.shape)

        # Positional embedding layer adds the positional embedding to the word embedding
        # Output shape: batch_size x words_per_sentence x d_model
        out = word_embeddings + self.Pe[torch.arange(word_embeddings.shape[1])]
        if(self.verbose):
            print("Positional embedding layer output shape: ", out.shape)
        return out