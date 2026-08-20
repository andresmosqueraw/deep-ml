import numpy as np

def accuracy_score(y_true, y_pred):
	correct_pred = 0
	for i in range(len(y_true)):
		if y_true[i] == y_pred[i]:
			correct_pred += 1
	
	return correct_pred / len(y_pred)