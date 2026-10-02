import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	rows_a = len(a)
	cols_a = len(a[0])

	rows_new = new_shape[0]
	cols_new = new_shape[1]

	if (rows_a * cols_a) != (rows_new*cols_new):
		return []

	flatten = []
	for row in a:
		flatten.extend(row)
	
	reshape_matrix = [[0] * cols_new for _ in range(rows_new)]

	counter = 0
	for i in range(rows_new):
		for j in range(cols_new):
			reshape_matrix[i][j] = flatten[counter]
			counter += 1

	return reshape_matrix