from itertools import product

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class IsingModel:

    def __init__(self, size = (100, 100),
                 temperature = 0.5,
                 initial_state = 'random',
                 periodic = False,
                 all_to_all_couplings = False,
                 coupling_strength = 1.0):
        self.shape = size
        self.dimension = len(size)
        # self.lin_size = size[0]
        self.temperature = temperature
        self.periodic = periodic

        if initial_state == 'random':
            self.spins = 2 * np.random.randint(0, 1 + 1, size = self.shape, dtype = np.int8) - 1
        elif initial_state == 'up':
            self.spins = np.ones(size = self.shape, dtype = np.int8)
        else:
            self.spins = np.ones(size = self.shape, dtype = np.int8)

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
            self.couplings = -1 * coupling_strength * np.ones(s, dtype = np.float)  # all ferromagnetic

        # Magnetic field at each point
        self.fields = np.zeros(self.shape, dtype = np.float)

    def get_random_index(self):
        """
        Returns a random index in the spin lattice
        """
        return [np.random.randint(0, dim) for dim in self.shape]

    def get_energy_at_site(self, position):
        """
        Compute the energy at a given site in the lattice.
        """

        e = 0.0

        pos = np.array(position)

        # Get the neighboring spins of pos
        for ax in range(self.dimension):
            # Make offset vector for axis like [0 0 0 ... 0 1 0 0]
            offset = np.zeros_like(pos)
            offset[ax] = 1
            # Add the "left" neighbor
            if (pos - offset)[ax] >= 0:
                e += self.spins[tuple(pos)] * self.spins[tuple(pos - offset)] * self.couplings[ax][tuple(pos - offset)]
            # Add the "right" neighbor
            if (pos + offset)[ax] < self.shape[ax]:
                e += self.spins[tuple(pos)] * self.spins[tuple(pos + offset)] * \
                     self.couplings[ax][tuple(pos)]  # no plus offset here
            # TODO: handle periodic case

        # Add contribution from field
        e += self.spins[tuple(pos)] * self.fields[tuple(pos)]

        return e

    def get_energy_at_site_2d(self, i, j):
        """
        Compute the energy at a given site in the lattice. Special optimized version for 2D case
        """

        if 1 <= i < self.shape[0] - 1 and 1 <= j < self.shape[1] - 1:
            return self.spins[i, j] * (
                    self.spins[i + 1, j] * self.couplings[0, i, j] +
                    self.spins[i - 1, j] * self.couplings[0, i - 1, j] +
                    self.spins[i, j + 1] * self.couplings[1, i, j] +
                    self.spins[i, j - 1] * self.couplings[1, i, j - 1] +
                    self.fields[i, j]
            )

        else:
            e = self.spins[i, j] * self.fields[i, j]
            if i < self.shape[0] - 1:
                e += self.spins[i, j] * self.spins[i + 1, j] * self.couplings[0, i, j]
            if i >= 1:
                e += self.spins[i, j] * self.spins[i - 1, j] * self.couplings[0, i - 1, j]
            if j < self.shape[1] - 1:
                e += self.spins[i, j] * self.spins[i, j + 1] * self.couplings[1, i, j]
            if j >= 1:
                e += self.spins[i, j] * self.spins[i, j - 1] * self.couplings[1, i, j - 1]
            return e

    def get_total_energy(self):
        """
        Gets the total internal energy of the lattice system, not normalized by number of spins
        """
        ranges = [range(x) for x in self.shape]
        all_indices = product(*ranges)

        return sum(self.get_energy_at_site(pos) for pos in all_indices)

    def metropolis_step(self):
        """
        Runs one step of the Metropolis-Hastings algorithm
        :return:
        """

        # Randomly select a site on the lattice
        # pos = self.get_random_index()
        i, j = [np.random.randint(0, dim) for dim in self.shape]

        # Calculate energy of the spin and energy if it is flipped
        # energy = self.get_energy_at_site(pos)
        energy = self.get_energy_at_site_2d(i, j)

        energy_flipped = -1 * energy

        # Flip the spin if it is energetically favorable. If not, flip based on Boltzmann factor
        if energy_flipped <= 0:
            self.spins[i, j] *= -1
        elif np.exp(-energy_flipped / self.temperature) > np.random.rand():
            self.spins[i, j] *= -1

    def run(self, epochs, video = False, show_progress = False):

        iterator = tqdm(range(epochs)) if show_progress else range(epochs)

        if video:
            num_frames = 100
            FFMpegWriter = anim.writers['ffmpeg']
            writer = FFMpegWriter(fps = 10)

            plt.ion()
            fig = plt.figure()

            with writer.saving(fig, "ising.mp4", 100):
                for epoch in iterator:
                    self.metropolis_step()
                    if epoch % (epochs // num_frames) == 0:
                        img = plt.imshow(self.spins, interpolation = 'nearest')
                        writer.grab_frame()
                        img.remove()

            plt.close('all')


        else:
            for epoch in iterator:
                self.metropolis_step()


if __name__ == "__main__":
    lattice = IsingModel((100, 100))
    lattice.run(1000000, video = True, show_progress = True)
