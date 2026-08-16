
import numpy as np

def r_squared(y_true, y_pred):
	sum_squared_residuals = np.sum((y_true - y_pred)**2)
	mean_actual_values = np.mean(y_true)
	total_sum_squares = np.sum((y_true - mean_actual_values)**2)
	return 1 - (sum_squared_residuals/total_sum_squares)
