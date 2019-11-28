from ising_compiler.ising_numpy import IsingModel
from ising_compiler.utils import *


class IsingGate:
    # footprint = ()
    num_inputs = 0
    num_outputs = 0

    def __init__(self, circuit=None, spins = (), inputs = (), outputs = ()):
        self.circuit = circuit
        self.spins = spins
        self.inputs = inputs
        self.outputs = outputs
        self._apply()

    def _apply(self):
        raise NotImplementedError

    def evaluate(self, inputs):
        raise NotImplementedError


class WIRE(IsingGate):

    # footprint = (2,1)
    num_inputs = 1
    num_outputs = 1
    wire_coupling = -1/2

    def _apply(self):
        i, o = self.inputs[0], self.outputs[0]
        self.circuit.set_coupling(i, o, value = self.wire_coupling)

    def evaluate(self, inputs):
        return inputs

class NAND(IsingGate):

    num_inputs = 2
    num_outputs = 1

    def _apply(self):
        pass



class IsingCircuit(IsingModel):
    def __init__(self, size = (100, 100),
                 temperature = 0.5,
                 initial_state = 'random'):
        super().__init__(size = size,
                         temperature = temperature,
                         initial_state = initial_state,
                         periodic = False,
                         all_to_all_couplings = False,
                         coupling_strength = 0.0)

    def get_coupling_index(self, s1, s2):
        '''Compute the indices corresponding to the coupling between neighboring spins s1 and s2'''
        assert is_adjacent(s1, s2), f"Error trying to get coupling index between non-adjacent spins {s1} and {s2}"
        offset = np.array(s2) - np.array(s1)
        offset_dir = np.nonzero(offset)
        offset_amt = np.sum(offset)
        assert len(offset_dir) == 1 and offset_dir[0] < self.dimension and np.abs(offset_amt) == 1.0

        offset_dir = np.sum(offset_dir)

        if offset_amt == 1:  # s2 is the "right" neighbor
            return (offset_dir,) + s1
        elif offset_amt == -1:  # s2 is the "left" neighbor
            return (offset_dir,) + s2
        else:
            raise ValueError()

    def set_coupling(self, s1, s2, value = 0.0):
        '''Set the coupling value between neighboring spins s1 and s2. -1 = ferromagnetic, +1 = antiferromagnetic'''
        indices = self.get_coupling_index(s1, s2)
        self.couplings[indices] = value

    def set_field(self, s, value=0.0):
        '''Set the magnetic field applied to spin s'''
        self.fields[s] = value

    def set_spin(self, s, value=False):
        '''Sets a specified spin to be up or down by applying a strong magnetic field'''
        FIELD_STRENGTH = 10.0
        if type(value) is bool:
            self.set_field(s, value=(FIELD_STRENGTH if value else -FIELD_STRENGTH))
        elif type(value) is int:
            if value == -1 or value == 0:
                self.set_field(s, value=-FIELD_STRENGTH)
            elif value == 1:
                self.set_field(s, value=FIELD_STRENGTH)
            else:
                raise ValueError()
        else:
            raise ValueError()

    # def WIRE(self, s1, s2):
    #     '''Copy the spin at s1 to s2'''
    #     assert is_adjacent(s1, s2)
    #     self.set_coupling(s1, s2, -1 / 2)
    #
    # def OR(self, in1, in2, out):
    #     '''OR gate, outputted to a third spin'''
    #     pass
    #
    # def XOR(self, in1, in2, ancilla, out):
    #     '''OR gate, outputted to a third spin'''
    #     pass
