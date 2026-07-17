
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


try:
    from GPTModel.GPTModel import GPTModel
    from GPTModel.LayerNorm import LayerNorm
    from GPTModel.TransformerBlock import TransformerBlock
   
except ImportError:
    from GPTModel.GPTModel import GPTModel
    from GPTModel.LayerNorm import LayerNorm
    from GPTModel.TransformerBlock import TransformerBlock
import tiktoken

from Tokenization.Embeddings import GPTDataLoaderV1


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    ##add batch dimension
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor
        
        
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def create_test_train_data_loaders():
    file_path = os.path.join(PROJECT_ROOT, "the-verdict.txt")
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    torch.manual_seed(123)
    _, train_loader = GPTDataLoaderV1.create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0 
    )
    
    _, val_loader = GPTDataLoaderV1.create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0 
    )
    
    return train_loader,val_loader

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader)==0:
        return float ("nan")
    elif num_batches  is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss = calc_loss_batch(
                input_batch=input_batch, target_batch=target_batch,model=model,device=device
            )
            total_loss +=loss.item()
        else:
            break
        
        
    return total_loss/num_batches


def main():
    
    train_loader, val_loader =  create_test_train_data_loaders()
    
    model = GPTModel(GPT_CONFIG_124M) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    with torch.no_grad():
        train_loss = calc_loss_loader(data_loader=train_loader, model=model, device=device)
        val_loss = calc_loss_loader(data_loader=val_loader, model=model, device=device)
        
    print("Training Loss::", train_loss)
    print("Validation Loss::", val_loss)
    

if __name__=="__main__":
    main()
    