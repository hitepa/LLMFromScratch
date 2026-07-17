import torch
import tiktoken

class SelfAttentionv1:


    def _createcontextvectorTest():
        inputs = torch.tensor(
        [[0.43,0.15,0.89],
         [0.55,0.87,0.66],
         [0.57,0.85,0.64],
         [0.22,0.58,0.33]]
        )
        ##Create context vector for second word in the sentence
        query = inputs[1]
        print()
        attention_scores_2 = torch.empty(inputs.shape[0])
        for i, x_i in enumerate(inputs):
            attention_scores_2[i] = torch.dot(x_i, query)
        print(attention_scores_2)
    
    def _createcontextvectorTestv2():
        inputs = torch.tensor(
        [[0.43,0.15,0.89],
         [0.55,0.87,0.66],
         [0.57,0.85,0.64],
         [0.22,0.58,0.33]]
        )
        ##Use faster matrix multiplication
        ##[4 X 3 ] @ [3 X 4] = [4 X 4]
        ##attention_scores matrix shape = [4x4]
        attention_scores = inputs @ inputs.T
        attention_weights = torch.softmax(attention_scores, dim=-1) 
        ##[4x4] @ [4x3]  = [4x3]
        all_context_vecs = attention_weights  @ inputs
        return all_context_vecs
    
    def _createContextVectorWithTrainableWeights():
         ##Input embeddings for sentence: Your journey starts at one step
         inputs = torch.tensor(
        [[0.43,0.15,0.89],
         [0.55,0.87,0.66],
         [0.57,0.85,0.64],
         [0.22,0.58,0.33],
         [0.77,0.25,0.10],
         [0.05,0.80,0.55]]
        )
         ##Define trainable weight matrices for Self-Attention Layer
         dim_in = inputs.shape[1]
         dim_out = 2
         ##Weight Matrix shape's now of rows must be equal to no of columns in inputs
         ##in order to make matrix multiplication possible
         torch.manual_seed(123)
         ##Shape = 3X2
         W_query = torch.nn.Parameter(torch.rand(dim_in, dim_out),requires_grad=False)
         W_key = torch.nn.Parameter(torch.rand(dim_in, dim_out),requires_grad=False)
         W_value = torch.nn.Parameter(torch.rand(dim_in, dim_out),requires_grad=False)

         ##Lets create attention score for second word
         ##We need q for second word, k for all words, v for all words
         ##Grab embeddings for second word and it query vector
         x_2 = inputs[1]
         ##1X3 @ 3X2 = 1X2
         query_2 = x_2@W_query
         print(query_2)

         ##Get K anv v vectors for all words
         ##6X3 @ 3X2 = 6X2
         keys = inputs@W_key
         values = inputs@W_value
         print("keys.shape::",keys.shape)
         print("values.shape::",values.shape)

         ##Lets calculate attention score of second word with second word
         ##We need q for second word, k for second word and v for second word
         keys_2 = keys[1]
         ##1X3  
         attn_score_22 = query_2.dot(keys_2)
         print("Attention score 22::",attn_score_22)
        
        ##1X2 @ 2X6 = 1X6
         attn_score_2 = query_2@ keys.T
         print("Unscaled Attention scores for second word::",+attn_score_2)

         d_k = keys.shape[-1]
         attn_weights_2 = torch.softmax(attn_score_2 / d_k**0.5 , dim = -1)
         print("Scaled Attention Weights::",attn_weights_2)

def main():
    SelfAttentionv1._createContextVectorWithTrainableWeights()
    ##print(all_context_vecs)


if __name__ == "__main__":
    main()



