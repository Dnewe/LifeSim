


def print_brain_vars(values):
    for a, weights in values.items():
        weight_dict = {k: round(v, 2) for k,v in weights.items()}
        print(f'{a}: {weight_dict}')