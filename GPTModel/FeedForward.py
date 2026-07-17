import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257, #Vocabulary size
    "context_length":1024, #Context_length
    "emb_dim":768, #Embedding vector size
    "n_heads":12, #Number of MHA
    "n_layers":12,
    "drop_rate":0.1, #drop rate to remove certain attention scores
    "qkv_bias": False #Add bias vector in q,k, v or not  
}

class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"]*4), 
            nn.GELU(),
            nn.Linear(cfg["emb_dim"]*4, cfg["emb_dim"])
        )
        
    def forward(self,x):
        return self.layers(x)
        
    
    
def main():
    
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2,3,GPT_CONFIG_124M["emb_dim"])
    out= ffn(x)
    print(out.shape)
    
if __name__=="__main__":
    main()        
        

        
        
        