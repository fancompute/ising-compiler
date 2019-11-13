from itertools import product

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

DEVICE = torch.device("cpu")


class IsingModel:

    def __init__(self, size = (100, 100),
                 temperature = 0.5,
                 initial_state = 'random',
                 periodic = False,
                 all_to_all_couplings = False):
        self.size = size
        self.dimension = len(size)
        # self.lin_size = size[0]
        self.temperature = temperature
        self.periodic = periodic

        if initial_state == 'random':
            self.spins = 2 * torch.randint(0, 1 + 1, size = self.size, dtype = torch.int8) - 1
        else:
            self.spins = torch.ones(size = self.size, dtype = torch.int8)

        if all_to_all_couplings:
            # TODO: use torch.sparse for this
            raise NotImplementedError()

        else:
            edges_per_spin = self.dimension  # in a square lattice
            s = (edges_per_spin,) + size
            # if self.periodic:
            #     s = (edges_per_spin,) + size
            # else:
            #     s = (edges_per_spin,) + tuple([d - 1 for d in self.size])
            # Couplings are a [dimension, n_x, n_y ...] tuple
            self.couplings = -1 * torch.ones(size = s, dtype = torch.float)  # all ferromagnetic

        # Magnetic field at each point
        self.fields = torch.zeros(size = self.size, dtype = torch.float)

    def get_random_index(self):
        """
        Returns a random index in the spin lattice
        """
        return [np.random.randint(0, dim) for dim in self.size]

    def get_energy_at_site(self, position):
        """
        Compute the energy at a given site in the lattice.
        """

        e = 0.0

        pos = torch.tensor(position)

        # Get the neighboring spins of pos
        for ax in range(self.dimension):
            # Make offset vector for axis like [0 0 0 ... 0 1 0 0]
            offset = torch.zeros_like(pos)
            offset[ax] = 1
            # Add the "left" neighbor
            if (pos - offset)[ax] >= 0:
                # print((pos - offset))
                # print(pos, offset, self.spins[pos], self.spins[pos - offset], self.couplings[axis][pos - offset])
                e += self.spins[tuple(pos)] * self.spins[tuple(pos - offset)] * self.couplings[ax][tuple(pos - offset)]
            # Add the "right" neighbor
            if (pos + offset)[ax] < self.size[ax]:
                # print((pos + offset))
                e += self.spins[tuple(pos)] * self.spins[tuple(pos + offset)] * \
                     self.couplings[ax][tuple(pos)]  # no plus offset here
            # TODO: handle periodic case

        # Add contribution from field
        e += self.spins[tuple(pos)] * self.fields[tuple(pos)]

        return e

    def get_total_energy(self):
        """
        Gets the total internal energy of the lattice system, not normalized by number of spins
        """
        ranges = [range(x) for x in self.size]
        all_indices = product(*ranges)

        return sum(self.get_energy_at_site(pos) for pos in all_indices)

    def run(self, epochs, num_frames = 100, video = True):

        FFMpegWriter = anim.writers['ffmpeg']
        writer = FFMpegWriter(fps = 10)

        plt.ion()
        fig = plt.figure()

        with writer.saving(fig, "ising.mp4", 100):
            for epoch in tqdm(range(epochs)):
                # Randomly select a site on the lattice
                pos = self.get_random_index()

                # Calculate energy of the spin and energy if it is flipped
                energy = self.get_energy_at_site(pos)
                energy_flipped = -1 * energy

                # Flip the spin if it is energetically favorable. If not, flip based on Boltzmann factor
                if energy_flipped <= 0:
                    self.spins[pos] *= -1
                elif np.exp(-energy_flipped / self.temperature) > np.random.rand():
                    self.spins[pos] *= -1

                if epoch % (epochs // num_frames) == 0:
                    if video:
                        img = plt.imshow(self.spins.numpy(), interpolation = 'nearest')
                        writer.grab_frame()
                        img.remove()

        # tqdm.write("Net Magnetization: {:.2f}".format(self.magnetization))

        plt.close('all')


if __name__ == "__main__":
    lattice = IsingModel((5, 5))
    lattice.run(1000000)
