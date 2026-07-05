def print_args(args):
    """Print all experiment arguments in two aligned columns."""
    items = sorted(vars(args).items())
    for i in range(0, len(items), 2):
        left_k, left_v = items[i]
        cell_l = f'  {left_k + ":":<22}{str(left_v):<26}'
        if i + 1 < len(items):
            right_k, right_v = items[i + 1]
            cell_r = f'{right_k + ":":<22}{str(right_v):<26}'
        else:
            cell_r = ''
        print(cell_l + cell_r)
