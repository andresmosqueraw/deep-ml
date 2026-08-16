import numpy as np

def cosine_similarity(v1, v2):
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	if len(v1) != len(v2):
		return -1
	
	v1 = np.array(v1)
	v2 = np.array(v2)

	l2_a = np.sqrt(np.sum(v1**2))
	l2_b = np.sqrt(np.sum(v2**2))

	return v1.dot(v2) / (l2_a * l2_b)