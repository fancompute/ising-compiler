import unittest

import numpy as np
from tqdm import tqdm

from ising_compiler.alu_nx import IsingCircuitModules
from ising_compiler.gates_nx import IsingCircuitGraph


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
        circuit.OUTPUT(c)

        all_input_combos = [[int(x) for x in ('{:02b}').format(n)] for n in range(2 ** 2)]

        for inputs in tqdm(all_input_combos, desc = gate_fn.__name__):
            expectations = circuit.evaluate_expectations({"A": inputs[0], "B": inputs[1]},
                                                         runs = 100,
                                                         epochs_per_run = 1000,
                                                         anneal_temperature_range = [.5, 1e-4],
                                                         show_progress = False)
            c_exp = expectations["C"]
            c_ideal = float(test_fn(inputs[0], inputs[1]))
            self.assertEqual(c_exp, c_ideal)

    def test_half_adder(self):
        circuit = IsingCircuitModules()
        A = circuit.INPUT("A")
        B = circuit.INPUT("B")
        S, C = circuit.HALF_ADDER(A, B, "S", "C")
        circuit.OUTPUT(S)
        circuit.OUTPUT(C)

        all_input_combos = [[int(x) for x in ('{:02b}').format(n)] for n in range(2 ** 2)]


        for inputs in tqdm(all_input_combos, desc = "HALFADDR"):
            a, b = inputs

            expectations = circuit.evaluate_expectations({"A": a, "B": b},
                                                         runs = 200,
                                                         epochs_per_run = 1000,
                                                         anneal_temperature_range = [.5, 1e-4],
                                                         show_progress = False)
            s_exp = expectations[S]
            cout_exp = expectations[C]

            self.assertEqual(a + b, s_exp + 2 * cout_exp)#, places=2)

    def test_full_adder(self):
        circuit = IsingCircuitModules()
        A = circuit.INPUT("A")
        B = circuit.INPUT("B")
        Cin = circuit.INPUT("Cin")
        S, Cout = circuit.FULL_ADDER(A, B, Cin, "S", "Cout")
        circuit.OUTPUT(S)
        circuit.OUTPUT(Cout)

        all_input_combos = [[int(x) for x in ('{:03b}').format(n)] for n in range(2 ** 3)]


        for inputs in tqdm(all_input_combos, desc = "FULLADDR"):
            a, b, cin = inputs

            expectations = circuit.evaluate_expectations({"A": a, "B": b, "Cin": cin},
                                                         runs = 100,
                                                         epochs_per_run = 10000,
                                                         anneal_temperature_range = [1, 1e-2],
                                                         show_progress = False)
            s_exp = expectations[S]
            cout_exp = expectations[Cout]

            self.assertAlmostEqual(a + b + cin, s_exp + 2 * cout_exp, places=2)


if __name__ == '__main__':
    unittest.main()
