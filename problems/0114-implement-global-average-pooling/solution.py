import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
	res = []
	# print(x.shape)
	n_rows = len(x)
	n_cols = len(x[0])
	n_channels = len(x[0][0])

	# for col in range(n_cols):
	# 	average = 0
	# 	for row in range(n_rows):
	# 		average += x[row][col]
	# 		print("x[row][col]: ", x[row][col], " average: ", average)
	# 	print("average: ", average, " rows: ", n_rows)
	# 	res.append(average / n_rows)
	
	for k in range(n_channels):
		sum_local = 0
		for c in range(n_cols):
			for r in range(n_rows):
				sum_local += x[r][c][k]
				# print("x[r][c][k]: ", x[r][c][k])
		# print("sum_local: ", sum_local)
		res.append(sum_local/(n_rows*n_cols))
	
	# print(res)
	return np.array(res)