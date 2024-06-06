
def print_dict(d, level=0, sep_key=':'):
    # Prints a dictionary
    for k, v in d.items():
        if isinstance(v, dict):
            print(level * '\t', k, sep_key)
            print_dict(v, level + 1)
        else:
            print(level * '\t', k, sep_key, v)


def print_dict_mean_value(d, level=0, print_value_dist=20):
    # Prints a dictionary
    for k, v in d.items():
        if isinstance(v, dict):
            print(level * '\t', k, ':')
            print_dict_mean_value(v, level + 1)
        else:
            v = np.array(v)
            v[np.isinf(v)] = 0
            white_space_string = (print_value_dist - len(k)) * ' '
            print(level * '\t', k, ':', white_space_string, np.mean(v))


def print_dict_collection(d, level=0, print_value_dist=20):
    # Prints a dictionary
    for k, v in d.items():
        if isinstance(v, dict):
            print(level * '\t', k, ':')
            print_dict_collection(v, level + 1)
        else:
            v = np.array(v)
            v[np.isinf(v)] = 0
            white_space_string = (print_value_dist - len(k)) * ' '
            print('\n' + level * '\t', k, ':')
            print_dict(collections.Counter(v), sep_key='x')

