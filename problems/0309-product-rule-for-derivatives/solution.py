import torch
import torch.nn.functional as F

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> torch.Tensor:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        torch.Tensor of coefficients of (f*g)' rounded to 4 decimal places
    """
    len_f = len(f_coeffs)
    len_g = len(g_coeffs)
    convolution = torch.zeros(len_f + len_g - 1)
    for i in range(len_f):
        for j in range(len_g):
            convolution[i+j] += f_coeffs[i] * g_coeffs[j]

    if len(convolution) == 1:
        return torch.tensor(0.0)

    return torch.arange(1, len(convolution)) * convolution[1:]