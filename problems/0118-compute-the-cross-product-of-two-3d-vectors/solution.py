import numpy as np

def cross_product(a, b):
    n = len(a)

    result = []
    for i in range(n):
        idx1 = (i + 1) % n
        idx2 = (i + 2) % n

        result.append(a[idx1]*b[idx2] - a[idx2]*b[idx1])
    
    return result