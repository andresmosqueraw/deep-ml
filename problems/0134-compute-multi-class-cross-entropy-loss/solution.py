import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    
    n_classes = len(predicted_probs)
    n_samples = len(predicted_probs[0])
    multi_ce = 0

    for i in range(n_classes):
        for j in range(n_samples):
            if true_labels[i][j] == 1:
                pred = predicted_probs[i][j]

                if pred < epsilon:
                    pred = epsilon
                elif pred > 1 - epsilon:
                    pred = 1 - epsilon
                
                multi_ce += np.log(pred)
    
    return -multi_ce / n_classes