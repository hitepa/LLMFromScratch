import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

##Data set class to convert raw text into array of input tokens and
##output tokens
class GPTDataSetV1(Dataset):
    def __init__(self,txt, tokenizer,max_length,stride):
        self.input_ids = []
        self.target_ids =[]
        self.vocab_size = tokenizer.n_vocab
        ##Tokenize the entire text
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1: i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
     ##Called internally by Data Loader to create batches
    def __getitem__(self,idx):
        return self.input_ids[idx], self.target_ids[idx]
    
class GPTDataLoaderV1:
    @staticmethod
    def create_dataloader_v1(txt, batch_size, max_length,
                             stride,shuffle,drop_last,num_workers):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDataSetV1(txt, tokenizer, max_length,stride)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, drop_last=drop_last,
                                num_workers=num_workers)
        return dataset, dataloader
    
class EmbeddingsLayer:
   def create_embeddings(vocab_size, output_dim):
      torch.manual_seed(123)
      ##tensor of shape vocab_size X output_dim
      embeddings = torch.nn.Embedding(vocab_size,output_dim)
      return embeddings

   def create_pos_embeddings(context_len, output_dim):
      pos_embeddings_layer = torch.nn.Embedding(context_len, output_dim)
      ##tensor of shape max_length X output_dim
      pos_embeddings = pos_embeddings_layer(torch.arange(context_len)) 
      return pos_embeddings
   
   


def main():
    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    ##Hyperparameters
    max_length = 4
    output_dim = 3

    ##Build the dataset and dataloader
    dataset, dataloader = GPTDataLoaderV1.create_dataloader_v1(
        raw_text, batch_size=1, max_length=max_length, stride=max_length,
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
    ##print("Token embeddings shape:", input_embeddings.shape)

    ##Positional embeddings
    pos_embeddings = EmbeddingsLayer.create_pos_embeddings(max_length, output_dim)
    ##print("Positional embeddings shape:", pos_embeddings.shape)

    ##Final input embeddings
    final_embeddings = input_embeddings + pos_embeddings
    print("Final input embeddings", final_embeddings)
    ##print("Final input embeddings shape:", final_embeddings.shape)


if __name__ == "__main__":
    main()