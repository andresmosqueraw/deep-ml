import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    t_max = t.max(dim=dim, keepdim=True).values
    exp_t = torch.exp(t - t_max)
    return exp_t / exp_t.sum(dim=dim, keepdim=True)
