import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps=1e-5
        self.scale= nn.Parameter(torch.ones(emb_dim))
        self.shift= nn.Parameter(torch.zeros(emb_dim))
        
        
        
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x=(x-mean) / (torch.sqrt(var+self.eps))
        return self.scale*norm_x + self.shift


def main():
    torch.manual_seed(123)
    torch.set_printoptions(sci_mode=False)
    batch_example = torch.randn(2, 5)
    layer_norm = LayerNorm(emb_dim=5)
    out = layer_norm(batch_example)

    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True, unbiased=False)

    print("Output:\n", out)
    print("Mean:\n", mean)
    print("Variance:\n", var)


if __name__ == "__main__":
    main()
