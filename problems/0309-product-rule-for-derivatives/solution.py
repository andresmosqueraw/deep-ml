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
    
    # convolucion
    product = [0.0] * (len(f_coeffs)*len(g_coeffs))
    for i, a in enumerate(f_coeffs):
        for j, b in enumerate(g_coeffs):
            product[i+j] += a * b
    
    # derivative
    derivative = []
    for k in range(1, len(product)):
        val = float(round(product[k] * k, 4))
        derivative.append(val)
    
    
    while len(derivative) > 0 and derivative[-1] == 0.0:
        derivative.pop()
    
    return derivative or [0.0]
    


