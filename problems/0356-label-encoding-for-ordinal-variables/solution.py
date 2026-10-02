def label_encode_ordinal(values: list, order: list) -> list:
    """
    Encode ordinal categorical values to integers based on specified order.
    
    Args:
        values: List of categorical values to encode
        order: List specifying the order of categories from lowest (0) to highest
    
    Returns:
        List of integers representing the encoded values
    """
    order_dict = {}
    for i in range(len(order)):
        order_dict[order[i]] = i

    res = []
    for v in values:
        if v not in order_dict:
            res.append(-1)
        else:
            res.append(order_dict[v])

    return res