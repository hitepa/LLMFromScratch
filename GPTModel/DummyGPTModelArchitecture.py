import torch
import torch.nn as nn
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257, #Vocabulary size
    "context_length":1024, #Context_length
    "emd_dim":768, #Embedding vector size
    "n_heads":12, #Number of MHA
    "n_layers":12,
    "drop_rate":0.1, #drop rate to remove certain attention scores
    "qkv_bias": False #Add bias vector in q,k, v or not  
}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emd_dim"])
        self.pos_embedding = nn.Embedding(cfg["context_length"],cfg["emd_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.out_head = nn.Linear(cfg["emd_dim"], cfg["vocab_size"], bias=False)
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg)
              for _ in range (cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emd_dim"])
        
        
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.token_embedding(in_idx)
        pos_embeds = self.pos_embedding(torch.arange(seq_len, device=in_idx.device))
        x = pos_embeds + tok_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
        
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
    def forward(self,x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self,normalized_shape, eps = 1e-5):
        super().__init__()
        
    def forward(self,x):
        return x


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Ever day holds a"
    
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    
    batch = torch.stack(batch, dim = 0)
    print(batch)
    
    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape::", logits.shape)
    print(logits)

if __name__ == "__main__":
    main()


           