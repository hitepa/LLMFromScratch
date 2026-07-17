import torch
import torch.nn as nn
import os
import sys
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from PreTraining import CrossEntropyLoss
from GPTModel.GPTModel import GPTModel, generate_text_sample,generate_text_sample_with_sampling

def train_model_simple(model,train_loader,val_loader,
                       optimizer,device,num_epochs,
                       eval_freq,eval_iter,start_context,tokenizer):
    train_losses, val_losses , track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    ##loop through epochs
    for epoch in range (num_epochs):
        model.train()
        ##In each epoch, calculate loss for all the batches
        ##loop through all batches in training data
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            ##calculate loss for a batch
            loss = CrossEntropyLoss.calc_loss_batch(input_batch=input_batch, target_batch=target_batch,
                                                    model=model, device=device)
            ##back propagate and adjust weight to reduce loss in the next batch
            loss.backward()
            optimizer.step()
            tokens_seen+=input_batch.numel()
            global_step+=1
            
            ##By now weights are adjusted, calculate validation loss and evaluate model accuracy
            ##validaton is performed over entire data set
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader,device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1}  (Step {global_step:06d}) :"
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")
                
    
    ##After all epochs, lets test model and generate text
    generate_and_print_sample(
        model, tokenizer, device, start_context
    ) 
    return train_losses, val_losses, track_tokens_seen



def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    ##disable gradient tracking and dropout
    with torch.no_grad():
        train_loss = CrossEntropyLoss.calc_loss_loader(
            model=model, data_loader=train_loader, device=device,num_batches=eval_iter
        )
        val_loss = CrossEntropyLoss.calc_loss_loader(
            model=model, data_loader=val_loader, device=device,num_batches=eval_iter
        )   
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    ##Pos embedding shape = context_size * emb_dim
    context_size = model.pos_emb.weight.shape[0]
    encoded = CrossEntropyLoss.text_to_token_ids(text=start_context, tokenizer=tokenizer).to(device=device)
    with torch.no_grad():
        token_ids = generate_text_sample(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )
        
        sampled_token_ids = generate_text_sample_with_sampling(model=model, idx=encoded,max_new_tokens=15,
                                                           context_size=GPT_CONFIG_124M["context_length"],
                                                           top_k=25,
                                                           temperature=1.4)
    
    decoded_text = CrossEntropyLoss.token_ids_to_text(token_ids=sampled_token_ids, tokenizer=tokenizer)
    print(f"Generated text:\n{decoded_text}")  
    model.train()
    
    
def main():
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )
    train_loader, val_loader = CrossEntropyLoss.create_test_train_data_loaders()
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=device, optimizer=optimizer,num_epochs=num_epochs,eval_freq=5,eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    ##After training, save the model weights (and optimizer state to resume later)
    save_path = os.path.join(PROJECT_ROOT, "PreTraining", "testLLM_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, save_path)
    print(f"Model saved to {save_path}")


if __name__=="__main__":
    main()
    
        
    
           