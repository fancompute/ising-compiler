import numpy as np

from ising_compiler.ising import IsingModel
from ising_compiler.utils import *


class IsingComputer(IsingModel):
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
        assert is_adjacent(s1, s2)
        offset = np.array(s2) - np.array(s1)
        offset_dir = np.nonzero(offset)
        offset_amt = np.sum(offset)
        assert len(offset_dir) == 1 and offset_dir[0] < self.dimension and np.abs(offset_amt) == 1.0

        offset_dir = np.sum(offset_dir)

        if offset_amt == 1: # s2 is the "right" neighbor
            return (offset_dir,) + s1
        elif offset_amt == -1: # s2 is the "left" neighbor
            return (offset_dir,) + s2
        else:
            raise ValueError()

    def set_coupling(self, s1, s2, value=0.0):
        '''Set the coupling value between neighboring spins s1 and s2. -1 = ferromagnetic, +1 = antiferromagnetic'''
        indices = self.get_coupling_index(s1, s2)
        self.couplings[indices] = value

    def WIRE(self, s1, s2):
        '''Copy the spin at s1 to s2'''
        assert is_adjacent(s1, s2)
        self.set_coupling(s1, s2, -1/2)

    def OR(self, in1, in2, out):
        '''OR gate, outputted to a third spin'''
        pass

    def XOR(self, in1, in2, ancilla, out):
        '''OR gate, outputted to a third spin'''
        pass
