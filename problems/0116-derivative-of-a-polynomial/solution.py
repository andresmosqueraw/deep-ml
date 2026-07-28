def poly_term_derivative(c: float, x: float, n: float) -> float:
    # Your code here
    # c * x^n = c * n * x ^ n-1
    return c * n * (x**(n-1))