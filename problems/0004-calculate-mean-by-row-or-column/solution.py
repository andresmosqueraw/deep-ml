def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if mode == 'row':
		return [sum(row) / len(row) for row in matrix]
	elif mode == 'column':
		return [sum(col) / len(col) for col in zip(*matrix)]
	return means