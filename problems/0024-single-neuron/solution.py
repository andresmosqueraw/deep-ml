import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    probabilities = []
    for x in features:
        z = sum(w * xi for w, xi in zip(weights, x)) + bias
        prob = 1 / (1 + math.exp(-z))
        probabilities.append(round(prob, 4))
    mse = sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)
    return probabilities, round(mse, 4)