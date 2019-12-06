import unittest
from json import loads

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

            self.assertEqual(a + b, s_exp + 2 * cout_exp)  # , places=2)

    def test_full_adder(self):
        self._test_full_adder_circuits(use_nand_adder = False)

    def test_full_adder_nand(self):
        self._test_full_adder_circuits(use_nand_adder = True)

    def _test_full_adder_circuits(self, use_nand_adder = False):
        circuit = IsingCircuitModules()
        A = circuit.INPUT("A")
        B = circuit.INPUT("B")
        Cin = circuit.INPUT("Cin")
        if use_nand_adder:
            S, Cout = circuit.FULL_ADDER_NAND(A, B, Cin, S = "S", Cout = "Cout")
            print("Testing full adder (NAND construction)...\n")
        else:
            S, Cout = circuit.FULL_ADDER(A, B, Cin, S = "S", Cout = "Cout")
            print("Testing full adder...\n")

        circuit.OUTPUT(S)
        circuit.OUTPUT(Cout)

        all_input_combos = [[int(x) for x in ('{:03b}').format(n)] for n in range(2 ** 3)]

        for inputs in all_input_combos:
            a, b, cin = inputs
            input_dict = {"A": a, "B": b, "Cin": cin}
            print("Testing with inputs {}".format(input_dict))
            outcomes = circuit.evaluate_outcomes(input_dict,
                                                 runs = 20,
                                                 epochs_per_run = 200000,
                                                 anneal_temperature_range = [1, 1e-4],
                                                 show_progress = True)

            most_common = outcomes.most_common()[0]
            most_common_outcome, most_common_frequency = most_common
            most_common_outcome = loads(most_common_outcome)
            desired_outcome = {"S"   : (a + b + cin) % 2,
                               "Cout": (a + b + cin) // 2}

            # most common outcome needs to be the correct one
            self.assertEqual(most_common_outcome, desired_outcome, msg = "Most common outcome is not desired one")

            # fail if below some normalized frequency
            total_trials = sum([tup[1] for tup in outcomes.most_common()])
            correct_rate = most_common_frequency / total_trials

            print("Accuraccy rate: {:.2f}".format(correct_rate))

            ACCURACCY_THRESHOLD = 0.75

            self.assertGreaterEqual(correct_rate, ACCURACCY_THRESHOLD, msg = "Accuraccy threshold not met")

    def test_ripple_carry_adder(self, num_bits = 4, num_trials = 3):
        circuit = IsingCircuitModules()
        S_bits, Cout = circuit.RIPPLE_CARRY_ADDER(num_bits)

        for _ in range(num_trials):
            num1, num2 = np.random.randint(0, 2 ** num_bits, size = 2)
            digs1 = [int(x) for x in ('{:0' + str(num_bits) + 'b}').format(num1)]
            digs2 = [int(x) for x in ('{:0' + str(num_bits) + 'b}').format(num2)]

            input_dict = {}
            for i, (a, b) in enumerate(zip(reversed(digs1), reversed(digs2))):
                input_dict["A" + str(i)] = a
                input_dict["B" + str(i)] = b

            digs_sum = [int(x) for x in ('{:0' + str(num_bits+1) + 'b}').format(num1+num2)]
            desired_output = {"C"+str(num_bits): digs_sum[0]}
            for i, dig in enumerate(reversed(digs_sum[1:])):
                desired_output["S"+str(i)] = dig

            print("Testing {} + {} = {} with inputs {}".format(num1, num2, num1 + num2, input_dict))
            print("Desired output: {}".format(desired_output))

            epochs_per_run = 100000 * 2 ** num_bits
            outcomes = circuit.evaluate_outcomes(input_dict,
                                                 runs = 10,
                                                 epochs_per_run = epochs_per_run,
                                                 anneal_temperature_range = [1, 1e-4],
                                                 show_progress = True)

            most_common = outcomes.most_common()[0]
            most_common_outcome, most_common_frequency = most_common
            most_common_outcome = loads(most_common_outcome)

            # most common outcome needs to be the correct one
            self.assertEqual(most_common_outcome, desired_output, msg = "Most common output is not desired one")

            # fail if below some normalized frequency
            total_trials = sum([tup[1] for tup in outcomes.most_common()])
            correct_rate = most_common_frequency / total_trials

            print("Accuraccy rate: {:.2f}".format(correct_rate))

            # ACCURACCY_THRESHOLD = 0.75
            # self.assertGreaterEqual(correct_rate, ACCURACCY_THRESHOLD, msg = "Accuraccy threshold not met")


if __name__ == '__main__':
    unittest.main()
