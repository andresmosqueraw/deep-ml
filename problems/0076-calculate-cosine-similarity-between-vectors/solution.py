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
	dot_product = v1 @ v2
	l2_v1 = np.sqrt(np.sum(v1**2))
	l2_v2 = np.sqrt(np.sum(v2**2))
	cosine_similarity = dot_product / (l2_v1*l2_v2)
	return cosine_similarity