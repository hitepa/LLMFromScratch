import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from LayerNorm import LayerNorm
    from TransformerBlock import TransformerBlock
except ImportError:
    from GPTModel.LayerNorm import LayerNorm
    from GPTModel.TransformerBlock import TransformerBlock
import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257, #Vocabulary size
    "context_length":1024, #Context_length
    "emb_dim":768, #Embedding vector size
    "n_heads":12, #Number of MHA
    "n_layers":12,
    "drop_rate":0.1, #drop rate to remove certain attention scores
    "qkv_bias": False #Add bias vector in q,k, v or not  
}

GPT_CONFIG_MEDIUM = {
    "vocab_size": 50257, #Vocabulary size
    "context_length":1024, #Context_length
    "emb_dim":1024, #Embedding vector size
    "n_heads":16, #Number of MHA
    "n_layers":24,
    "drop_rate":0.1, #drop rate to remove certain attention scores
    "qkv_bias": False #Add bias vector in q,k, v or not  
}

GPT_CONFIG_LARGE = {
    "vocab_size": 50257, #Vocabulary size
    "context_length":1024, #Context_length
    "emb_dim":1280, #Embedding vector size
    "n_heads":20, #Number of MHA
    "n_layers":36,
    "drop_rate":0.1, #drop rate to remove certain attention scores
    "qkv_bias": False #Add bias vector in q,k, v or not  
}


GPT_CONFIG_XL = {
    "vocab_size": 50257, #Vocabulary size
    "context_length":1024, #Context_length
    "emb_dim":1600, #Embedding vector size
    "n_heads":25, #Number of MHA
    "n_layers":48,
    "drop_rate":0.1, #drop rate to remove certain attention scores
    "qkv_bias": False #Add bias vector in q,k, v or not  
}




class GPTModel (nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg)  for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        
        ##Embeddings Layer
        tok_embeddings = self.tok_emb(in_idx)
        pos_embeddings = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeddings + pos_embeddings
        
        ##Dropout embeddings to avoid overfitting
        x = self.drop_emb(x)
        
        ##Transformer blocks, where each block contains a sequence of mha, dropout, norm , ffn, dropout
        x = self.trf_blocks(x)
        
        ##o/p layer: norm, linear logits
        x  = self.final_norm(x)
        logits = self .out_head(x)
        
        return logits
    
def generate_text_sample(model, idx, max_new_tokens, context_size):
    #idx shape = batch_size,seq_len
    for _ in range(max_new_tokens):
        ##From i/p grab last context_size number of tokens
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        ##logits shape = batch_size_n_token,vocab_size
        ##keep batch, choose last token, keep last dimension also    
        logits = logits[:,-1,:]
        probas = torch.softmax(logits, dim =-1)
        idx_next = torch.argmax(probas, dim  = -1, keepdim=True)
        idx  = torch.cat((idx,idx_next), dim =1)
        
    return idx

##Text generation using temperature and top_k sampling
def generate_text_sample_with_sampling(model, idx, max_new_tokens,
                                       context_size, temperature=0.0, top_k = None, eos_id=None):
    for _ in range (max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]
        
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:,-1]
            logits = torch.where(
                logits<min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
            
        if temperature>0.0:
            logits = logits/temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits,dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx,idx_next), dim=1)
        
    return idx
        
            
        
    
    
def main():
    
    ##test_model()
    test_generate_txt()
    
    
def test_model():
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Ever day holds a"
    
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    
    batch = torch.stack(batch, dim = 0)
    print(batch)
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape::", logits.shape)
    print(logits)
    
    ##Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the model: {total_params}")
    ##total ffn parameters in the model
    total_ffn_params = sum(p.numel() for p in model.trf_blocks[0].ff.parameters())
    print(f"Total parameters in the FeedForward layer of the first Transformer block: {total_ffn_params}")
    ##total mha parameters in the model
    mha_params = sum(p.numel() for p in model.trf_blocks[0].att.parameters())
    print(f"Parameters in the MultiHeadAttention layer of the first Transformer block: {mha_params}")
    
    ##model size
    model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming 4 bytes per parameter (float32)
    print(f"Approximate model size: {model_size_mb:.2f} MB")
    
    
    
def test_generate_txt():
    tokenizer = tiktoken.get_encoding("gpt2")
    start_context = "Every effort moves you"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor_shape:" , encoded_tensor.shape)   
    
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    out = generate_text_sample(model, encoded_tensor, max_new_tokens=6, context_size=encoded_tensor.shape[1])
    
    out_with_sampling = generate_text_sample_with_sampling(model=model, idx=encoded_tensor,max_new_tokens=15,
                                                           context_size=GPT_CONFIG_124M["context_length"],
                                                           top_k=25,
                                                           temperature=1.4)
    
    
    print("Generated text indices:", out_with_sampling)
    decoded = tokenizer.decode(out_with_sampling[0].tolist())
    print("Generated text:", decoded)
    
    
if __name__ == "__main__":
    main()