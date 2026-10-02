import numpy as np

def make_diagonal(x):
	len_x = len(x)
	diagonal = [[0] * len_x for _ in range(len_x)]
	
	for i in range(len_x):
		diagonal[i][i] = x[i]

	return diagonal