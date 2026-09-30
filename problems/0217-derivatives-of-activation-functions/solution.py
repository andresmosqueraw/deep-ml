import math
def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	sigmoid = 1 / (1+math.exp(-x))
	tanh = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
	return {
		'sigmoid': sigmoid * (1-sigmoid), 
		'tanh': 1 - tanh**2, 
		'relu': 1 if x > 0 else 0
	}