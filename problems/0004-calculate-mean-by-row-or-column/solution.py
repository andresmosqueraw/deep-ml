def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	means = []
	if mode == 'row':
		for row in matrix:
			means.append(sum(row)/len(row))
	elif mode == 'column':
		for i in range(len(matrix[0])):
			sum_vals = 0
			for j in range(len(matrix)):
				sum_vals += matrix[j][i]
			means.append(sum_vals/len(matrix))

	return means