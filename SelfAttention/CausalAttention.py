import os
import sys
import torch
import torch.nn as nn

##Add the project root (parent of this folder) to sys.path so the
##Tokenization package can be imported regardless of the current working dir
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Tokenization.Embeddings import EmbeddingsLayer
from Tokenization.Embeddings import GPTDataLoaderV1

class CausalAttention(nn.Module):
    def __init__(self, d_in,d_out,context_length,dropout,qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        ##Linear transform will create tensor of shape = d_outXd_in
        self.W_query = nn.Linear(d_in,d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', 
                             torch.triu(torch.ones(context_length,context_length),
                                        diagonal=1))
        

    def forward(self, x):
        ##X is batch of shape say: 2 X 6 X 3, where 2 is no of inputs in batch,
        ##6 is no ok tokens in each input, 3 is dimensionality of each token
        b, num_tokens, d_in = x.shape
        ##Assume d_out = 4
        ##2X6X3 . 4X3 = (2X6, 3) . 3X4 = 12X3@3X4 = 12X4
        ##Preserve batch and deflat stacked tokens, final shape = 2X6X4
        keys = self.W_key(x)
        query = self.W_query(x)
        values = self.W_value(x)
        ##(2X6X4) @ (2X  4X6) = 2 X 6 X 6
        attn_scores = query@keys.transpose(1,2)
        ##Mask attention scores along the diagonal
        attn_scores.masked_fill_(self.mask.bool() [:num_tokens, :num_tokens],-torch.inf)
        attn_weights = torch.softmax(attn_scores/keys.shape[-1] **0.5, dim =1)
        attn_weights = self.dropout(attn_weights)
        ##2X6X6 @ 2X6X4 = 2X6X4
        context_vecs = attn_weights @ values
        return context_vecs

def main():
    ##the-verdict.txt lives at the project root, resolve it relative to this file
    file_path = os.path.join(PROJECT_ROOT, "the-verdict.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    ##Hyperparameters
    max_length = 6
    output_dim = 3
    batch_size = 2

    ##Build the dataset and dataloader
    dataset, dataloader = GPTDataLoaderV1.create_dataloader_v1(
        raw_text, batch_size=batch_size, max_length=max_length, stride=max_length,
        shuffle=False, drop_last=True, num_workers=0)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs Token ID:", inputs)
    print("Targets Token ID:", targets)
    ##print("Inputs shape:", inputs.shape)
    ##print("Targets shape:", targets.shape)

    ##Token embeddings
    token_embeddings = EmbeddingsLayer.create_embeddings(dataset.vocab_size, output_dim)
    input_embeddings = token_embeddings(inputs)
    print("Input embeddings", input_embeddings)
    print("Token embeddings shape:", input_embeddings.shape)

    ##Positional embeddings
    pos_embeddings = EmbeddingsLayer.create_pos_embeddings(max_length, output_dim)
    print("Positional embeddings shape:", pos_embeddings.shape)

    ##Final input embeddings
    final_embeddings = input_embeddings + pos_embeddings
    print("Final input embeddings shape:", final_embeddings.shape)
    print("Final input embeddings", final_embeddings)
    ##Call CausalAttention
    torch.manual_seed(123)
    d_in = final_embeddings.shape[-1]  ##embedding size = 4
    d_out = 2                ##projected Q/K/V size     
    context_length = max_length
    dropout = 0.1
    qkv_bias = False
    causal_attention = CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
    ce = causal_attention(final_embeddings)
    print("Causal Attention output shape", ce.shape)
    print("Causal Attention output", ce)


if __name__ == "__main__":
    main()


