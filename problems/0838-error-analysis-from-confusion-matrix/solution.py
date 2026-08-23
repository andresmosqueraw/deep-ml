import numpy as np

def error_analysis(cm):
    """
    Analyze errors in a multi-class confusion matrix.

    Args:
        cm: n x n confusion matrix (list of lists or numpy array),
            cm[i][j] = count of true class i predicted as j.

    Returns:
        Dict with keys 'per_class_error_rate', 'overall_error_rate',
        'most_confused_pair', 'worst_class'.
    """
    row_sum_i = np.sum(cm, axis=1)
    diagonal = np.array([cm[i][i] for i in range(len(cm))])
    per_class_error_rate = (row_sum_i - diagonal) / row_sum_i

    accuracy = np.sum(diagonal) / np.sum(cm)
    overall_error_rate = 1.0 - accuracy

    most_confused_pair = [0, 1]
    max_confused = 0
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            if i != j:
                if cm[i][j] > max_confused:
                    most_confused_pair = [i, j]
                    max_confused = cm[i][j]

    worst_class = np.argmax(per_class_error_rate)

    return {"per_class_error_rate": per_class_error_rate.tolist(), 
    "overall_error_rate": overall_error_rate, 
    "most_confused_pair": most_confused_pair, 
    "worst_class": worst_class}


