import sys
import os
import torch
import torch.nn as nn
import tiktoken

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Tokenization.Embeddings import EmbeddingsLayer
from Tokenization.Embeddings import GPTDataLoaderV1

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads ==0) , "d_out must be divisible by num_heads"

        ##d_out: Projected output dimension from weight matrix
        self.d_out = d_out
        self.num_heads= num_heads
        ##Dimension of each head
        self.head_dim = d_out//num_heads
        ##Assume d_in = 3, d_out = 6.
        ##In Linear transformation shape of weight matrix will be 6X3
        self.W_query = nn.Linear(d_in, d_out, qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, qkv_bias)
        ##Dont know what is this
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length),diagonal=1))

    
    def forward(self,x):
        ##Assume shape of input tensor as 1X4X3 (batchXtokensXd_in)
        ##1X4X3
        b,num_tokens,d_in = x.shape
        ##Shapes of key, query and value tensor: 1X4X6: Input Dimension projected from 3-D to 6-D
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        ##Split vector from shape 1X4X6 (bXnum_tokensXd_out) to tensor of shape 1X4X2X3 (bXnum_tokensXnum_headsXhead_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        ##Need to transpose to bring num_head before and create shape of: (bXnum_headsXnum_tokensXXhead_dim)
        ##This will fix batch and num_heads and we can perform matrix multplication between tokens and dimension accross each batch and each head
        ##Matrix multiplication in Pytorch is performed on last two dimensions of the tensor. So we need to bring num_heads before num_tokens and head_dim
        ##Tensor shape now is:: batch X num_Heads X num_tokens X head_dimension
        ##For our example:: 1 X 2 X 4 X 3
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)
        
        ##Attention score shape: 1X2X4X4
        ##Calcuating relation of each token with other token for each batch and for each head under each batch
        attn_scores = queries@keys.transpose(2,3)

        ##Cuasal Attention
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        ##Self_Attention and softmax
        attn_weights = torch.softmax(attn_scores/keys.shape[-1] ** 0.5, dim=-1)

        ## (1x2x4x4) @ (1x2x4x3) = (1x2x4x3)
        context_vec = attn_weights @ values

        ##transpose to 1x4x2x3 
        context_vec = context_vec.transpose(1,2)

        ##Deflate 
        context_vec = context_vec.contiguous().view(b, num_tokens,self.d_out)
        ##Final linear projection
        context_vec = self.out_proj(context_vec)
        return context_vec


def main():
    ##the-verdict.txt lives at the project root, resolve it relative to this file
    file_path = os.path.join(PROJECT_ROOT, "the-verdict.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    ##Hyperparameters
    max_length = 6
    output_dim = 2
    batch_size = 2

    ##Build the dataset and dataloader
    dataset, dataloader = GPTDataLoaderV1.create_dataloader_v1(
        raw_text, batch_size=batch_size, max_length=max_length, stride=max_length,
        shuffle=False, drop_last=True, num_workers=0)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    ##print("Inputs Token ID:", inputs)
    ##print("Targets Token ID:", targets)
    ##print("Inputs shape:", inputs.shape)
    ##print("Targets shape:", targets.shape)

    ##Token embeddings
    token_embeddings = EmbeddingsLayer.create_embeddings(dataset.vocab_size, output_dim)
    input_embeddings = token_embeddings(inputs)
    ##print("Input embeddings", input_embeddings)
    ##print("Token embeddings shape:", input_embeddings.shape)

    ##Positional embeddings
    pos_embeddings = EmbeddingsLayer.create_pos_embeddings(max_length, output_dim)
    ##print("Positional embeddings shape:", pos_embeddings.shape)

    ##Final input embeddings
    final_embeddings = input_embeddings + pos_embeddings
    print("Final input embeddings shape:", final_embeddings.shape)
    print("Final input embeddings", final_embeddings)
    
    
   
    ##Test Batch For MultiHeadAttention Starts
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
         [0.55, 0.87, 0.66],
         [0.57, 0.85, 0.64],
         [0.22, 0.58, 0.33],
         [0.77, 0.25, 0.10],
         [0.05, 0.80, 0.55]]
    )
    test_batch = torch.stack([inputs, inputs], dim=0)
    batch_size, context_length,d_in = test_batch.shape
    print(test_batch.shape)
    ##Test Batch For MultiHeadAttention Ends
    
    
 #####################---------------------------------------------################################ 
 
 
   
    ##Actual batch for MultiHeadAttention Starts
    ##batch = final_embeddings
    ##batch_size, context_length,d_in = batch.shape
    ##Actual batch for MultiHeadAttention Ends
    
    
    ##Call MultiHeadAttention
    torch.manual_seed(123)
    d_out = 2
    num_heads = 2
    mha = MultiHeadAttention(d_in=d_in, d_out=d_out, context_length=context_length,dropout=0, num_heads=num_heads,qkv_bias=False)
    context_vec = mha.forward(test_batch)
    print(context_vec)
    
    

if __name__ == "__main__":
    main()
    
    



        


