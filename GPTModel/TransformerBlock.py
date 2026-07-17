import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SelfAttention.MultiHeadAttention import MultiHeadAttention
try:
    from LayerNorm import LayerNorm
    from FeedForward import FeedForward
except ImportError:
    from GPTModel.LayerNorm import LayerNorm
    from GPTModel.FeedForward import FeedForward

GPT_CONFIG_124M = {
    "vocab_size": 50257, #Vocabulary size
    "context_length":1024, #Context_length
    "emb_dim":768, #Embedding vector size
    "n_heads":12, #Number of MHA
    "n_layers":12,
    "drop_rate":0.1, #drop rate to remove certain attention scores
    "qkv_bias": False #Add bias vector in q,k, v or not  
}

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att= MultiHeadAttention(
         d_in=cfg["emb_dim"],
         d_out=cfg["emb_dim"],
         context_length=cfg["context_length"],
         num_heads=cfg["n_heads"],
         dropout=cfg["drop_rate"],
         qkv_bias=cfg["qkv_bias"])
        self.ff=FeedForward(cfg)
        self.norm1=LayerNorm(cfg["emb_dim"])
        self.norm2=LayerNorm(cfg["emb_dim"])
        self.drop_shortcut= nn.Dropout(cfg["drop_rate"])
        
        
    def forward(self, x):
        shortcut =x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
    
    

def main():
    torch.manual_seed(123)
    x = torch.rand(2, 4, GPT_CONFIG_124M["emb_dim"])
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()

