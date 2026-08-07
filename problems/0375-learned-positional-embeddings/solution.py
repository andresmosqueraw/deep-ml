import numpy as np

def learned_positional_encoding(token_embeddings: np.ndarray, position_embedding_table: np.ndarray,
                                start_pos: int = 0) -> np.ndarray:
    token_embeddings = np.array(token_embeddings, dtype=float)
    position_embedding_table = np.array(position_embedding_table, dtype=float)

    # Cuantos tokens tiene cada secuencia
    seq_len = token_embeddings.shape[1]

    # 1. Sacar de la tabla las filas correspondientes a las posiciones que necesitamos
    #    Desde start_pos hasta start_pos + seq_len (sin incluir el final)
    positional_embeddings = position_embedding_table[start_pos : start_pos + seq_len]
    # shape resultante: (seq_len, d_model)

    # 2. Sumar los embeddings posicionales a cada token
    #    Broadcasting: (batch_size, seq_len, d_model) + (seq_len, d_model)
    #    NumPy repite automaticamente la suma para cada sample del batch
    result = token_embeddings + positional_embeddings

    return result