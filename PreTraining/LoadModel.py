
import torch
import torch.nn as nn
import os
import sys
import tiktoken


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    

from GPTModel.GPTModel import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257, #Vocabulary size
    "context_length":256, #Context_length
    "emb_dim":768, #Embedding vector size
    "n_heads":12, #Number of MHA
    "n_layers":12,
    "drop_rate":0.1, #drop rate to remove certain attention scores
    "qkv_bias": False #Add bias vector in q,k, v or not  
}



def main():
    checkpoint = torch.load("")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()

