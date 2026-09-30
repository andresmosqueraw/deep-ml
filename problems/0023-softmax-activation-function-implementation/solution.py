import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    scores_tensor = torch.tensor(scores, dtype=torch.float)
    softmax = torch.softmax(scores_tensor, dim=0)
    return softmax.tolist()