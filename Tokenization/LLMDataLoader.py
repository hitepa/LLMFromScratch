import torch
##print(torch.cuda.is_available())
from torch.utils.data import Dataset, DataLoader
import tiktoken

##Data Set to create a dataset for GPT model training
class GPTDatasetV1(Dataset):
    ##In constructor initialize input and target tensors by using entire tokens created from entire data set
    def __init__(self, txt, tokenizer, max_length,stride):
        self.input_ids = []
        self.target_ids = []
        ##Token IDs for entire text, Passed encoder is BPE
        token_ids = tokenizer.encode(txt)
        self.vocab_size = tokenizer.n_vocab
        self.max_length = len(token_ids)
        

        ##Loop through entire tokens and create training data set i.e inputs_ids and target_ids tensor

        for i in range (0 , len(token_ids)-max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            output_chunk = token_ids[i+1 : i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(output_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    

    def __getitem__(self, idx):
        return self.input_ids[idx] , self.target_ids[idx]
    
    def __vocab_size__(self):
        return self.vocab_size
    
    def __max_length__(self):
        return self.max_length
    



##Data Loader to create batches from the GPT dataset
class GPTDataLoaderV1:
    @staticmethod
    def create_dataloader_v1(txt, batch_size, max_length,stride,shuffle,drop_last,num_workers):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(txt, tokenizer,max_length, stride)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                shuffle=shuffle,drop_last=drop_last,
                                num_workers=num_workers)
        
        return dataloader


def main():
    file_path = "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    dataloader = GPTDataLoaderV1.create_dataloader_v1(raw_text, batch_size=2, max_length=4, stride=1, shuffle=False, drop_last=True, num_workers=0)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)


if __name__ == "__main__":
    main()
