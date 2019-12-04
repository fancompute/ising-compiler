import unittest

from ising_compiler.gates_nx import *


class GateTests(unittest.TestCase):

    def test_gates(self):
        gates = [IsingCircuitGraph.AND,
                 IsingCircuitGraph.NAND,
                 IsingCircuitGraph.OR,
                 IsingCircuitGraph.NOR,
                 IsingCircuitGraph.XOR,
                 IsingCircuitGraph.XNOR]
        tests = [lambda a, b: a & b,
                 lambda a, b: not (a & b),
                 lambda a, b: a | b,
                 lambda a, b: not (a | b),
                 lambda a, b: a ^ b,
                 lambda a, b: not (a ^ b)]

        for gate, test in zip(gates, tests):
            self._test_gate_2in_1out(gate, test)

    def _test_gate_2in_1out(self, gate_fn, test_fn):

        circuit = IsingCircuitGraph()
        a = circuit.INPUT("A")
        b = circuit.INPUT("B")
        c = gate_fn(circuit, a, b, "C")
        c = circuit.OUTPUT(c)

        all_input_combos = [[int(x) for x in ('{:0' + str(2) + 'b}').format(n)] for n in range(2 ** 2)]

        for inputs in tqdm(all_input_combos, desc = gate_fn.__name__):
            expectations = circuit.evaluate_expectations({"A": inputs[0], "B": inputs[1]},
                                                         runs = 100,
                                                         epochs_per_run = 1000,
                                                         anneal_temperature_range = [.5, 1e-3],
                                                         show_progress = False)
            c_exp = expectations["C"]
            c_ideal = float(test_fn(inputs[0], inputs[1]))
            self.assertEqual(c_exp, c_ideal)


if __name__ == '__main__':
    unittest.main()
