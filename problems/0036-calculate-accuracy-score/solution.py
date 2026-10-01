import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
	TP_and_TN = sum(1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_pred)))
	# print(TP_and_TN)
	total = len(y_true)
	accuracy = TP_and_TN / total
	return accuracy