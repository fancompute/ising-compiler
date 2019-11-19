import unittest

from tqdm import tqdm

from ising_compiler.gates import *
from ising_compiler.utils import *


class GateTests(unittest.TestCase):

    def test_couplings(self):
        size = (5, 5)
        for _ in tqdm(range(100), desc = "Testing coupling get/set"):
            coupling = 1 if np.random.rand() < .5 else -1
            c = IsingCircuit(size = size, initial_state = 'random', temperature = 1e-4)
            s1 = get_random_site(size)
            s2 = get_random_neighbor(s1, lattice_size = size)
            c.set_coupling(s1, s2, value = coupling)
            c.run(500, video = False, show_progress = False)
            spin1 = c.spins[s1]
            spin2 = c.spins[s2]
            assert spin1 == -1 * coupling * spin2

    def test_gates(self):
        gates = [WIRE]
        tests = [self._test_WIRE]

        for gate, test in zip(gates, tests):
            self._test_gate(gate, test)

    def _test_gate(self, gate_cls, test_fn, num_trials=100):
        all_input_combos = [[int(x) for x in ('{:0' + str(gate_cls.num_inputs) + 'b}').format(1)]
                            for n in range(2 ** gate_cls.num_inputs)]

        for _ in tqdm(range(100), desc = "Testing {}".format(gate_cls.__name__)):
            for inputs in all_input_combos:
                test_fn(inputs)

    def _test_WIRE(self, inputs):
        size = (5, 5)
        c = IsingCircuit(size = size, initial_state = 'random', temperature = 1e-4)
        s1 = get_random_site(size)
        s2 = get_random_neighbor(s1, lattice_size = size)
        c.set_spin(s1, value = inputs[0])
        WIRE(circuit = c, spins=(s1,s2), inputs=(s1,), outputs=(s2,))
        c.run(500)
        assert c.spins[s1] == c.spins[s2]


if __name__ == '__main__':
    unittest.main()
