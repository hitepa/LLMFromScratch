
import torch
import torch.nn as nn
import os
import sys
import numpy as np
import tiktoken

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    


from GPTModel.GPTModel import GPTModel, generate_text_sample,generate_text_sample_with_sampling
from PreTraining import CrossEntropyLoss
from CrossEntropyLoss import text_to_token_ids, token_ids_to_text

GPT_CONFIG_124M = {
    "vocab_size": 50257, #Vocabulary size
    "context_length":1024, #Context_length
    "emb_dim":768, #Embedding vector size
    "n_heads":12, #Number of MHA
    "n_layers":12,
    "drop_rate":0.1, #drop rate to remove certain attention scores
    "qkv_bias": False #Add bias vector in q,k, v or not  
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim":768, "n_layers":12, "n_heads":12},
    "gpt2-medium (355M)": {"emb_dim":1024, "n_layers":24, "n_heads":16},
    "gpt2-large (774M)": {"emb_dim":1280, "n_layers":36, "n_heads":20},
    "gpt2-xl (1558M)": {"emb_dim":1600, "n_layers":48, "n_heads":25},
}

import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch05/"
       "01_main-chapter-code/gpt_download.py"
       )
filename = url.split('/')[-1]
urllib.request.urlretrieve(url,filename=filename)

def main():
    from gpt_download import download_and_load_gpt2
    settings, params = download_and_load_gpt2(
        model_size="124M", models_dir="gpt2"
    )
    
    model_name = "gpt2-small (124M)"
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    
    ##Align model with original config used in GPT-2
    NEW_CONFIG.update({"qkv_bias": True})
    
    ##This has default weights
    gpt = GPTModel(NEW_CONFIG)
    gpt.eval()

    ##Load the pretrained OpenAI weights into the model
    load_weights_into_gpt(gpt, params)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    gpt.to(device=device)
    torch.manual_seed(123)
    token_ids = generate_text_sample_with_sampling(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer=tokenizer).to(device=device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
        )
    
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    
    
    
    
def assign(left,right):
    if left.shape !=right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                         f"Right: {right.shape}")
        
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt,params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight , params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight , params['wte'])
    
    ##Assign weights in transformer blocks
    for b in range(len(params ["blocks"])):
        ##Update attention weights of q, k and v matrices
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis = -1)
        gpt.trf_blocks[b].att.W_query.weight = assign( gpt.trf_blocks[b].att.W_query.weight,q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign( gpt.trf_blocks[b].att.W_key.weight,k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign( gpt.trf_blocks[b].att.W_value.weight,v_w.T)
        
        ##Update bias  of q, k and v matrices
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis = -1)
        gpt.trf_blocks[b].att.W_query.bias = assign( gpt.trf_blocks[b].att.W_query.bias,q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign( gpt.trf_blocks[b].att.W_key.bias,k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign( gpt.trf_blocks[b].att.W_value.bias,v_b)
        
        
        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, 
        params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, 
        params["blocks"][b]["attn"]["c_proj"]["b"])
        
        ##Update weight and bias of FFN
        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight,params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias,params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight,params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias,params["blocks"][b]["mlp"]["c_proj"]["b"])
        
        ##Update weight and bias of Layer Norm
        gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale,params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift,params["blocks"][b]["ln_1"]["b"]) 
        gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale,params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift,params["blocks"][b]["ln_2"]["b"])
    
    ##Update weight and bias of fial Layer Norm whic sits outside transformer block
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
         
         
         
    
if __name__=="__main__":
    main()


    
    

