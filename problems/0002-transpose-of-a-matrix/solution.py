def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    # return [list(col) for col in zip(*a)]

    rows = len(a)
    cols = len(a[0])

    output = [[0] * rows for _ in range(cols)]
    for c in range(cols):
        for r in range(rows):
            output[c][r] = a[r][c] 
            # print(a[r][c])
    
    return output