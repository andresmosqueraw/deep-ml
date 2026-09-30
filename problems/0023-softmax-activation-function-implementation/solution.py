import math

def softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    exponents = [math.exp(s-max_score) for s in scores]
    sum_exponents = sum(exponents)
    softmax = [e / sum_exponents for e in exponents]
    return softmax