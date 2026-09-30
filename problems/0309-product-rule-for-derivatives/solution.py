import numpy as np

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """
    len_f = len(f_coeffs)
    len_g = len(g_coeffs)
    convolution = [0] * (len_f + len_g - 1)

    for i in range(len_f):
        for j in range(len_g):
            convolution[i+j] += f_coeffs[i] * g_coeffs[j]

    print(convolution)
    if len(convolution) == 1:
        return [0.0]
    derivatives = []
    for i in range(1, len(convolution)):
        derivatives.append(convolution[i] * i)

    return derivatives
