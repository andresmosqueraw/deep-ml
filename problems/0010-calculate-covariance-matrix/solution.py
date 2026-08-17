def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	n_features = len(vectors)
	n_observations = len(vectors[0])
	covariance_matrix = [[0.0] * n_features for _ in range(n_features)]

	means = [sum(feature) / n_observations for feature in vectors]

	for i in range(n_features):
		for j in range(i, n_features):
			total_sum = 0
			for k in range(n_observations):
				diff_i = vectors[i][k] - means[i]
				diff_j = vectors[j][k] - means[j]
				total_sum += diff_i * diff_j
			degrees_of_freedom = n_observations - 1
			covariance = total_sum / degrees_of_freedom
			covariance_matrix[i][j] = covariance_matrix[j][i] = covariance
		
	return covariance_matrix
			