import numpy as np


def is_adjacent(s1, s2):
    '''Returns whether spin sites s1 and s2 are neighbors'''
    return np.linalg.norm(np.array(s2) - np.array(s1)) == 1.0


def get_random_site(lattice_size):
    s = np.zeros(len(lattice_size), dtype = int)
    for i, max_dim in enumerate(lattice_size):
        s[i] = np.random.randint(0, max_dim)
    return tuple(s)


def get_random_neighbor(s1, lattice_size = None):
    dim = len(s1)
    offset = np.zeros(dim, dtype = int)
    offset_dim = np.random.randint(0, dim)
    # Set offset[offset_dim] equal to +1 or -1
    if np.random.rand() < 0.5:
        offset[offset_dim] = -1
    else:
        offset[offset_dim] = 1

    s2 = tuple(np.array(s1) + offset)

    if lattice_size is not None:
        if np.min(s2) < 0 or any(map(lambda x: x < 1, np.array(lattice_size) - np.array(s2))):
            # Invalid neighbor
            return get_random_neighbor(s1, lattice_size = lattice_size)

    return s2
