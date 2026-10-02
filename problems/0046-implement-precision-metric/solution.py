import numpy as np
def precision(y_true, y_pred):
	TP = np.sum((y_true == 1) & (y_pred == 1))
	FP = np.sum((y_true == 0) & (y_pred == 1))
	if TP + FP == 0:
		return 0
	return TP / (TP+FP)