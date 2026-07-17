import torch
import torch.nn as nn
import os
import sys

GPT_CONFIG_124M = {
    "vocab_size": 50257, #Vocabulary size
    "context_length":256, #Context_length
    "emd_dim":768, #Embedding vector size
    "n_heads":12, #Number of MHA
    "n_layers":12,
    "drop_rate":0.1, #drop rate to remove certain attention scores
    "qkv_bias": False #Add bias vector in q,k, v or not  
}

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from GPTModel.GPTModel import GPTModel
    from GPTModel.LayerNorm import LayerNorm
    from GPTModel.TransformerBlock import TransformerBlock
   
except ImportError:
    from GPTModel.GPTModel import GPTModel
    from GPTModel.LayerNorm import LayerNorm
    from GPTModel.TransformerBlock import TransformerBlock
import tiktoken


class TestCrossEntropyLoss(nn.Module):
    
    def text_to_token_ids(text, tokenizer):
        encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
        ##add batch dimension
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor
        
        
    def token_ids_to_text(token_ids, tokenizer):
        flat = token_ids.squeeze(0)
        return tokenizer.decode(flat.tolist())
    
    
    
    
def main():
    inputs = torch.tensor([[16833,3626,6100], ##every effort moves
                           [40,1107,588]])     ##i really like   
    
    ##these are desired o/p token ids
    
    targets = torch.tensor([[3626,6100,345], ##effort moves you
                           [1107,588,11311]])     ##really like chocolate
    
    model = GPTModel(GPT_CONFIG_124M) 
    
    with torch.no_grad():
        logits = model(inputs)
        ## Probability of all possible words in vocab
        probas = torch.softmax(logits, dim = -1)
        print("probs shape", probas.shape)
        
    print("logits shape::", logits.shape)
    print("targets shape::", targets.shape)
    
    ##Flatten to same shape
    logits_flat = logits.flatten(0,1)
    targets_flat = targets.flatten()
    print("flattened logits shape::", logits_flat.shape)
    print("flattened targets shape::", targets_flat.shape)
    
    ##generate probability, get probability of targets, take average of probabilities, multiply with -1
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    
    print(loss)
    
    
    
    
    
    
    
        
    
    
    
if __name__ == "__main__":
    main()

