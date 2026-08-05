import math

def softmax(scores: list[float]) -> list[float]:
    # Paso 1: Encontrar el valor máximo de la lista
    max_val = max(scores)
    
    # Paso 2: Calcular exp(xi - max) para cada elemento
    exps = [math.exp(i - max_val) for i in scores]
    
    # Paso 3: Sumar todas las exponenciales
    sum_exps = sum(exps)
    
    # Paso 4: Dividir cada exponencial entre la suma total
    return [e / sum_exps for e in exps]