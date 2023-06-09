from . import *

class LayerNorm(nn.Module):
    '''
    This is similar to nn.LayerNorm(axis = -1)
    '''
    def __init__(self, d_model, eps = 1e-12, verbose = False):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.verbose = verbose
        self.eps = eps

    def forward(self, x):
        # input shape: batch_size x words_per_sentence x d_model
        if(self.verbose):
            print("LayerNorm input shape: ", x.shape)
        normalized = (x-x.mean(dim=-1, keepdim=True))/((x + self.eps).std(dim=-1, correction=0, keepdim=True) )
        # scale and shift each dimension of d_model independently
        scaled_and_shifted = self.gamma * normalized + self.beta 
        # output shape=input shape
        if(self.verbose):
            print("LayerNorm input shape: ", scaled_and_shifted.shape)
        return scaled_and_shifted
