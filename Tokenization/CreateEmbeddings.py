##Create Embeddings for token IDs and also create positional encoded embedding for token ids
import torch
import tiktoken
from LLMDataLoader import GPTDatasetV1
from LLMDataLoader import GPTDataLoaderV1  

class EmbeddingsLayer:

    def create_bpe(self):
        bpe = tiktoken.get_encoding("gpt2")
        return bpe

    def load_sample_data(self):
        file_path = "the-verdict.txt"
        with open(file_path, "r", encoding="utf-8") as file:
            raw_text = file.read()
        return raw_text

    def create_data_loader(self, dataset, raw_text, bpe, max_length, stride, batch_size):
        data_loader = GPTDataLoaderV1.create_dataloader_v1(dataset, raw_text, batch_size, max_length, stride, shuffle=False,
                                                        drop_last=True, num_workers=0)
        return data_loader

    def create_batch(self):
        bpe = self.create_bpe()
        raw_text = self.load_sample_data()
        batch_size = 1
        max_length = 4
        stride = 4
        data_set = self.create_data_set(raw_text, bpe, max_length, stride)
        data_loader = self.create_data_loader(data_set, raw_text, bpe, max_length, stride, batch_size)
        data_iter = iter(data_loader)
        return data_iter, data_set,data_loader
    
    def create_embeddings(self, data_set, data_iter):
        token_embedding_layer = self.create_token_embeddings_layer(data_set)
        positional_embedding_layer = self.create_positional_embeddings_layer(data_set)
        inputs, target = next(data_iter)
        print("Input Shape:", inputs.shape)
        print("Target Shape:", target.shape)
        input_embeddings = token_embedding_layer(inputs)
        target_embeddings = token_embedding_layer(target)
        print("Input Embeddings Shape:", input_embeddings.shape)
        print("Target Embeddings Shape:", target_embeddings.shape)
        input_pos_embeddings = positional_embedding_layer(torch.arange(4))
        target_pos_embeddings = positional_embedding_layer(torch.arange(4)) 
        print("Input Pos Embeddings Shape:", input_pos_embeddings.shape)
        print("Target Pos Embeddings Shape:", target_pos_embeddings.shape)
        final_input_embeddings = input_embeddings + input_pos_embeddings
        final_target_embeddings = target_embeddings + target_pos_embeddings 
        print("Final Input Embeddings Shape:", final_input_embeddings.shape)
        print("Final Target Embeddings Shape:", final_target_embeddings.shape)
        return final_input_embeddings, final_target_embeddings

    def create_data_set(self, raw_text, bpe, max_length, stride):
        data_set = GPTDatasetV1(raw_text, bpe, max_length, stride)
        return data_set

    def create_token_embeddings_layer(self, data_set):
        vocab_size = data_set.__vocab_size__()
        embedding_size = 256
        token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)
        return token_embedding_layer
    
    def create_positional_embeddings_layer(self, data_set):
        context_size= 4
        embedding_size = 256
        positional_embedding_layer = torch.nn.Embedding(context_size, embedding_size)
        return positional_embedding_layer
    
def main():
    embedding_layer = EmbeddingsLayer()
    data_iter, data_set, data_loader = embedding_layer.create_batch()
    input_embeddings, target_embeddings = embedding_layer.create_embeddings(data_set, data_iter)
    print(input_embeddings)
    print(target_embeddings)


if __name__ == "__main__":
    main()

    





