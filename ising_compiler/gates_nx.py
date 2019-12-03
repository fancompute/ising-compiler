import networkx as nx
from tqdm import tqdm

from ising_compiler.ising_nx import IsingGraph
from ising_compiler.utils import *


class IsingCircuitGraph(IsingGraph):
    wire_coupling = -3.0
    input_field_strength = 10.0

    '''
    Class representing a graph which is sequentially built to emulate a Boolean circuit. Includes methods for adding
    various gates. By convention, all nodes which serve as inputs to the circuit or subcomponents are copied with wires,
    and all outputs are not. (e.g. NAND(A,B,C) will take inputs A' and B' and return spin C)
    '''

    def __init__(self, temperature = 0.5, initializer = 'random'):
        super().__init__(nx.Graph(), temperature = temperature, initializer = initializer)
        self.inputs = []
        self.outputs = []

    def get_spin(self, node, mode = 'spin'):
        if mode == 'spin':
            return self.graph.nodes[node]['spin']
        elif mode == 'bool':
            return True if self.graph.nodes[node]['spin'] == 1 else False
        elif mode == 'binary':
            return 1 if self.graph.nodes[node]['spin'] == 1 else 0
        else:
            raise ValueError()

    def add_spin(self, node, field_strength = 0.0):
        assert not self.graph.has_node(node), f"Node {node} already in graph {self.graph}"
        self.graph.add_node(node)
        # Initialize spin of new node
        if self.initializer == 'random':
            self.graph.nodes[node]['spin'] = 1 if np.random.rand() > .5 else -1
        elif self.initializer == 'up':
            self.graph.nodes[node]['spin'] = 1
        elif self.initializer == 'down':
            self.graph.nodes[node]['spin'] = -1
        self.graph.nodes[node]['field'] = field_strength
        return node

    def copy_inputs(self, *nodes):
        input_copies = []
        for node in nodes:
            assert self.graph.has_node(node), f"Input node {node} missing in graph {self.graph}"
            node_copy = self.add_spin(node + "'")
            input_copies.append(node_copy)
            self.WIRE(node, node_copy)
        return input_copies

    def add_spins_not_present(self, *nodes):
        for node in nodes:
            if not self.graph.has_node(node):
                self.add_spin(node)
        return nodes

    def set_coupling(self, spin1, spin2, coupling_strength):
        if not self.graph.has_edge(spin1, spin2):
            self.graph.add_edge(spin1, spin2)
        self.graph.edges[(spin1, spin2)]['coupling'] = coupling_strength

    def set_field(self, spin, field_strength):
        self.graph.nodes[spin]['field'] = float(field_strength)

    def set_input_fields(self, input_dict, mode = 'spin'):
        for input_spin, input_value in input_dict.items():
            if mode == 'bool':
                assert type(input_value) is bool
                self.set_field(input_spin, (-self.input_field_strength if input_value else self.input_field_strength))
            elif mode == 'spin':
                if input_value == -1:
                    self.set_field(input_spin, self.input_field_strength)
                elif input_value == 1:
                    self.set_field(input_spin, -self.input_field_strength)
                else:
                    raise ValueError()
            elif mode == 'binary':
                if input_value == 0:
                    self.set_field(input_spin, self.input_field_strength)
                elif input_value == 1:
                    self.set_field(input_spin, -self.input_field_strength)
                else:
                    raise ValueError()
            else:
                raise ValueError()

    def INPUT(self, name):
        '''Designate a spin to be an input to the circuit. Returns a wired *copy* of the spin node'''
        self.add_spin(name)
        self.inputs.append(name)
        return name

    def OUTPUT(self, name):
        assert self.graph.has_node(name), f"Output node {name} missing in graph {self.graph}"
        self.outputs.append(name)
        return name

    def WIRE(self, spin1, spin2):
        self.set_coupling(spin1, spin2, self.wire_coupling)

    def AND(self, in1, in2, out):
        s1, s2 = self.copy_inputs(in1, in2)
        s3 = self.add_spin(out)
        # H = -1/2 s1 + -1/2 s2 + s3 + 1/2 s1 s2 + -s1 s3 + -s2 s3
        self.set_field(s1, -1 / 2)
        self.set_field(s2, -1 / 2)
        self.set_field(s3, 1)
        self.set_coupling(s1, s2, 1 / 2)
        self.set_coupling(s1, s3, -1)
        self.set_coupling(s2, s3, -1)
        return s3

    def NAND(self, in1, in2, out):
        s1, s2 = self.copy_inputs(in1, in2)
        s3 = self.add_spin(out)
        # H = -1/2 s1 + -1/2 s2 + -s3 + 1/2 s1 s2 + s1 s3 + s2 s3
        self.set_field(s1, -1 / 2)
        self.set_field(s2, -1 / 2)
        self.set_field(s3, -1)
        self.set_coupling(s1, s2, 1 / 2)
        self.set_coupling(s1, s3, 1)
        self.set_coupling(s2, s3, 1)
        return s3

    def OR(self, in1, in2, out):
        s1, s2 = self.copy_inputs(in1, in2)
        s3 = self.add_spin(out)
        # H = -1/2 s1 + -1/2 s2 + -s3 + 1/2 s1 s2 + s1 s3 + s2 s3
        self.set_field(s1, -1 / 2)
        self.set_field(s2, -1 / 2)
        self.set_field(s3, -1)
        self.set_coupling(s1, s2, -1 / 2)
        self.set_coupling(s1, s3, -1)
        self.set_coupling(s2, s3, -1)
        return s3

    def NOR(self, in1, in2, out):
        s1, s2 = self.copy_inputs(in1, in2)
        s3 = self.add_spin(out)
        # H = -1/2 s1 + -1/2 s2 + -s3 + 1/2 s1 s2 + s1 s3 + s2 s3
        self.set_field(s1, 1 / 2)
        self.set_field(s2, 1 / 2)
        self.set_field(s3, 1)
        self.set_coupling(s1, s2, 1 / 2)
        self.set_coupling(s1, s3, 1)
        self.set_coupling(s2, s3, 1)
        return s3

    def XOR(self, in1, in2, out):
        s1, s2 = self.copy_inputs(in1, in2)
        so = self.add_spin(out)
        ancilla = str(len(self.graph.nodes))
        sA = self.add_spin(ancilla)
        # H = 1/2 s1 + 1/2 s2 + 1 sA + 1/2 so + 1/2 s1 s2 + s1 sA + s2 sA + 1/2 s1 so + 1/2 s2 so + sA so
        self.set_field(s1, 1 / 2)
        self.set_field(s2, 1 / 2)
        self.set_field(sA, 1)
        self.set_field(so, 1 / 2)
        self.set_coupling(s1, s2, 1 / 2)
        self.set_coupling(s1, sA, 1)
        self.set_coupling(s2, sA, 1)
        self.set_coupling(s1, so, 1 / 2)
        self.set_coupling(s2, so, 1 / 2)
        self.set_coupling(sA, so, 1)

        return so

    def XNOR(self, in1, in2, out):
        s1, s2 = self.copy_inputs(in1, in2)
        so = self.add_spin(out)
        ancilla = str(len(self.graph.nodes))
        sA = self.add_spin(ancilla)
        # H = 1/2 s1 + 1/2 s2 + 1 sA + 1/2 so + 1/2 s1 s2 + s1 sA + s2 sA + 1/2 s1 so + 1/2 s2 so + sA so
        self.set_field(s1, 1 / 2)
        self.set_field(s2, 1 / 2)
        self.set_field(sA, 1)
        self.set_field(so, -1 / 2)
        self.set_coupling(s1, s2, 1 / 2)
        self.set_coupling(s1, sA, 1)
        self.set_coupling(s2, sA, 1)
        self.set_coupling(s1, so, -1 / 2)
        self.set_coupling(s2, so, -1 / 2)
        self.set_coupling(sA, so, -1)

        return so

    def evaluate_input(self, input_dict,
                       epochs = 10000,
                       anneal_temperature_range = None,
                       show_progress = False,
                       mode = 'bool'):
        '''Evaluates the circuit one time for a given input'''
        # re-initialize all spins
        IsingGraph.initialize_spins(self.graph)
        # set the input fields to the input of the circuit
        self.set_input_fields(input_dict, mode = mode)
        # run metropolis / annealing
        self.run_metropolis(epochs, anneal_temperature_range = anneal_temperature_range, show_progress = show_progress)
        # build a return dictionary of output spins
        output_dict = {}
        for output in self.outputs:
            output_dict[output] = self.get_spin(output, mode = mode)
        return output_dict

    def evaluate_expectations(self, input_dict,
                              runs = 1000,
                              epochs_per_run = 1000,
                              anneal_temperature_range = None,
                              show_progress = True):
        '''Evaluates the expectation of output spins over many runs'''
        output_dicts = []
        iterator = tqdm(range(runs)) if show_progress else range(runs)
        for _ in iterator:
            output_dict = self.evaluate_input(input_dict,
                                              epochs = epochs_per_run,
                                              anneal_temperature_range = anneal_temperature_range,
                                              mode = 'binary',
                                              show_progress = False)
            output_dicts.append(output_dict)

        # Compute mean dictionary
        mean_dict = {}
        for key in output_dicts[0].keys():
            mean_dict[key] = sum(d[key] for d in output_dicts) / len(output_dicts)
        return mean_dict

    def evaluate_circuit(self,
                         runs = 1000,
                         epochs_per_run = 1000,
                         anneal_temperature_range = None,
                         show_progress = True):
        '''Evaluates the expectation of output spins over many runs for every possible combination of inputs'''
        all_input_combos = [[int(x) for x in ('{:0' + str(len(self.inputs)) + 'b}').format(n)]
                            for n in range(2 ** len(self.inputs))]
        all_output_dicts = {}

        iterator = tqdm(all_input_combos) if show_progress else all_input_combos
        for inputs in iterator:
            input_dict = {}
            for i, spin in enumerate(self.inputs):
                input_dict[spin] = inputs[i]
            all_output_dicts[str(input_dict)] = self.evaluate_expectations(input_dict,
                                                                           runs = runs,
                                                                           epochs_per_run = epochs_per_run,
                                                                           anneal_temperature_range = anneal_temperature_range,
                                                                           show_progress = False)

        return all_output_dicts
