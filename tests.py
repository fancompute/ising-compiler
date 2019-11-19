import unittest

from tqdm import tqdm

from ising_compiler.gates import IsingComputer
from ising_compiler.utils import *


class GateTests(unittest.TestCase):

    def test_couplings(self):
        size = (5, 5)
        for _ in tqdm(range(100), desc = "Testing coupling get/set"):
            coupling = 1 if np.random.rand() < .5 else -1
            c = IsingComputer(size = size, initial_state = 'random', temperature = 1e-4)
            s1 = get_random_site(size)
            s2 = get_random_neighbor(s1, lattice_size = size)
            c.set_coupling(s1, s2, value = coupling)
            c.run(500, video = False, show_progress = False)
            spin1 = c.spins[s1]
            spin2 = c.spins[s2]
            assert spin1 == -1 * coupling * spin2

    def test_WIRE(self):
        pass


if __name__ == '__main__':
    unittest.main()
