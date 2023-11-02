import torch


# OR table

Ys=torch.tensor([1, 1, 0, 1], dtype=torch.float32)

Xs=torch.tensor([
    [1, 1],
    [1, 0],
    [0, 0],
    [0, 1]
], dtype=torch.float32)

W=torch.zeros([1, 2], dtype=torch.float32).uniform_(0, 1)

b= torch.zeros((1, 1), dtype=torch.float32)

def perceptron (X : torch.Tensor, W : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(X @ W.T + b) 

def loss (Y : torch.Tensor, Y_: torch.Tensor) -> torch.Tensor:
    return((Y - Y_)**2).sum()
lr = 0.01



Y = (perceptron(Xs[None, :], W, b) > 0.5).float()
print (Y)

