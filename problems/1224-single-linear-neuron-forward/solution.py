import torch
import torch.nn as nn


def single_neuron_forward(x):
    """Forward pass of one fixed linear neuron.

    Args:
        x: torch.Tensor of shape (1, 3).

    Returns:
        Python float, the neuron output.
    """
    # 1. Crear la neurona: 3 entradas, 1 salida
    neuron = nn.Linear(3, 1)

    # 2. Fijar los pesos y el bias (sin trackear gradientes)
    with torch.no_grad():
        neuron.weight.copy_(torch.tensor([[0.5, -0.2, 0.3]]))
        neuron.bias.copy_(torch.tensor([0.1]))

    # 3. Correr el forward pass
    output = neuron(x)

    # 4. Convertir el tensor a float de Python
    return output.item()