from . import *

class SingleHeadAttention(nn.Module):
    '''
    Implements softmax(QK/root(d_k))V for a single attention head
    if d_k, etc. not provided, all of them must be equal to d_model
    '''
    def __init__(self, d_model, d_head, init_range=0.02, verbose = False, save_pattern=False):
        super().__init__()
        self.Wk = nn.Parameter(torch.empty(d_model, d_head))
        self.Wv = nn.Parameter(torch.empty(d_model, d_head))
        self.Wq = nn.Parameter(torch.empty(d_model, d_head))
        nn.init.normal_(self.Wk, mean=0, std=init_range)
        nn.init.normal_(self.Wv, mean=0, std=init_range)
        nn.init.normal_(self.Wq, mean=0, std=init_range)
        self.d_k = d_head
        self.verbose = verbose
        self.save_pattern = save_pattern
        self.pattern = None
    
    def forward(self, residual_stream):
        if(self.verbose):
            print("Attention head input shape: ", residual_stream.shape)
            print("Attention head Wk, Wv, Wq shape: ", self.Wk.shape) # d_model -> d_head
        # K, Q, V are mappings from resid stream to d_head dimensional vectors
        # So the matrices are low-rank (d_head < d_model)
        K =  residual_stream @ self.Wk 
        Q =  residual_stream @ self.Wq 
        V =  residual_stream @ self.Wv 

        if(self.verbose):
            print("Attention head Q, K, V shapes: ", Q.shape, K.shape, V.shape)
        out = self.attention(Q, K, V)
        if(self.verbose):
            print("Attention head output shape: ", out.shape)
        return out

    def attention(self, Q, K, V):
        # dims Q/K/V: batch_size x words_per_sentence x d_head
        # each word is now a weighted sum of the other words in the sentence
        QK = Q @ K.transpose(-2, -1) # [b, n, d] @ [b, d, n] -> [b, n, n] (n = words_per_sentence)
        # Make QK causal:
        # We need to use tril to remove 'future' Key vectors:
        # QK = [ [ Q[0]K[0], *remove : Q[0]K[1],.., Q[0]K[-1]* ], 
        #       [ Q[1]K[0], Q[1]K[1], * remove : .., Q[1]K[-1]* ], 
        #       ... ]
        mask = torch.tril(torch.ones(QK.shape)).to(QK.device)
        mask[mask == 0] = -float('inf')
        mask[mask == 1] = 0
        QK = torch.tril(QK) + mask
        pattern = nn.Softmax(dim=-1)( ( QK ) / ( self.d_k**.5 ) )
        if(self.save_pattern): 
            self.pattern = pattern
        # dims after softmax: batch_size x words_per_sentence x words_per_sentence
        return  pattern @ V
    
class MultiHeadAttention(nn.Module):
    '''
    Implements
    1. Delegate attention to attention heads
    2. Wo @ (Attention heads) 
    3. Write to residual stream
    '''
    def __init__(self, d_model, n_heads, init_range=0.02, verbose = False, save_pattern=False, parallel=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        d_head = int(d_model/n_heads)
        self.verbose = verbose
        if(not parallel):
            self.attention_heads = nn.ModuleList([
                SingleHeadAttention(d_model, 
                                    d_head,
                                    init_range, verbose, save_pattern=save_pattern) 
                        for _ in range(n_heads)])
        else:
            # parallel attention heads
            self.Wk = nn.Parameter(torch.empty(d_model, d_model))
            self.Wv = nn.Parameter(torch.empty(d_model, d_model))
            self.Wq = nn.Parameter(torch.empty(d_model, d_model))
            nn.init.normal_(self.Wk, mean=0, std=init_range)
            nn.init.normal_(self.Wv, mean=0, std=init_range)
            nn.init.normal_(self.Wq, mean=0, std=init_range)
            self.d_model = d_model

        self.Wo = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.normal_(self.Wo, mean=0, std=init_range)
        self.save_pattern = save_pattern
        self.pattern = None
        self.parallel = parallel
        
    def forward(self, residual_stream):
        if(self.verbose):
            print("Multi attention head input shape: ", residual_stream.shape)
        if not self.parallel:            
            attention_out_list = [self.attention_heads[i](residual_stream) 
                            for i in range(self.n_heads)] 
            attention_out = torch.cat(attention_out_list, dim=-1)
            if(self.save_pattern):
                self.pattern = [self.attention_heads[i].pattern for i in range(self.n_heads)]
            if(self.verbose):
                print("Multi attention head output shapes: ", [x.shape for x in attention_out_list])
                print("Multi attention head concatenated shape: ", attention_out.shape)
            
            # each word is now a weighted sum of the other words in the sentence
            out = attention_out @ self.Wo
            # write to residual stream
            out = residual_stream + out
            if(self.verbose):
                print("Multi attention head output shape: ", out.shape)
            return out
        else:
            wps = residual_stream.shape[1]
            batch_size = residual_stream.shape[0]
            qs = torch.einsum('bwm, mo -> bwo', 
                            residual_stream, self.Wq).reshape(
                                (batch_size, 
                                self.n_heads, 
                                wps, -1))
            vs = torch.einsum('bwm, mo -> bwo', 
                            residual_stream, self.Wv).reshape(
                                (batch_size, 
                                self.n_heads, 
                                wps, -1))
            ks = torch.einsum('bwm, mo -> bwo', 
                            residual_stream, self.Wk).reshape(
                                (batch_size, 
                                self.n_heads, 
                                wps, -1))
            qks = torch.einsum('...nd, ...od->...no', qs, ks) 
            mask = torch.tril(torch.ones_like(qks))
            mask[mask == 0] = -float('inf')
            mask[mask == 1] = 0
            qks = torch.tril(qks) + mask

            pattern = (nn.Softmax(dim=-1)( 
                ( qks ) / ((self.d_model / self.n_heads)**.5) 
                ))
            
            if(self.save_pattern):
                self.pattern = pattern
            
            out = torch.einsum('ij, ...j -> ...i',                  # matmul
                               self.Wo,                             # d_model x d_model
                               ((
                                   pattern @ vs                     # batch x n_heads x wps x d_head
                                ).reshape((batch_size, wps, -1)))   # batch x wps x d_model
                                )                                   # batch x wps x d_model
            
            if(self.verbose):
                print("[parallel] Multihead attention Q, K, V shapes: ", qs.shape, ks.shape, vs.shape)
                print("[parallel] Multihead attention Q @ K shape: ", qks.shape)
                print("[parallel] Multihead attention pattern shape: ", pattern.shape)
                print("[parallel] Multihead attention output shape: ", out.shape)

            return residual_stream + out