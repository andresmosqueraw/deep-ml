import numpy as np
import math

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    n_samples = len(predicted_probs)
    n_classes = len(predicted_probs[0])

    total_loss = 0.0

    for i in range(n_samples):
        sample_loss = 0.0
        for j in range(n_classes):
            prob = predicted_probs[i][j]
            label = true_labels[i][j]

            if prob < epsilon:
                prob = epsilon
            elif prob > 1 - epsilon:
                prob = 1 - epsilon
            
            sample_loss += label * math.log(prob)

        sample_loss = -sample_loss
        total_loss += sample_loss

    mean_loss = total_loss / n_samples
    return mean_loss