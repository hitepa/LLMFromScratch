import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self,d_in,d_out):
        ##d_in is dimension of input Embedding vector
        ##d_out is dimension of weight matrix. So essentially converting input embedding is dimension size d_in to d_out
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))


    def forward(self, x):
        query = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value
        
        ##Dot Product between Q and KT: Q.KT
        attn_scores = query @ keys.T
        ##Normalize Scale Dot Product: Softmax ( (Q.KT)/sqrt(Dk) )
        attn_weights = torch.softmax(attn_scores/ keys.shape[-1]**0.5, dim = -1)
        ##Weighted sum of attention weights: Softmax ( (Q.KT)/sqrt(Dk) ) X V
        context_vec = attn_weights @ values
        return context_vec


if __name__ == "__main__":
    torch.manual_seed(123)
    ##Input embeddings for: "Your journey starts at one step"
    ##6 tokens, each a 3-dimensional embedding -> shape (6, 3)
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
    )

    d_in = inputs.shape[1]   ##embedding size = 3
    d_out = 2                ##projected Q/K/V size

    sa = SelfAttention_v1(d_in, d_out)
    context_vecs = sa(inputs)

    print("Input shape:       ", inputs.shape)
    print("Context vec shape: ", context_vecs.shape)
    print(context_vecs)
