import matplotlib.animation as anim
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm


class IsingGraph:

    def __init__(self, graph, temperature = 0.5, initializer = 'random', coupling_strength = 1.0):

        assert type(graph) is nx.Graph

        if nx.get_node_attributes(graph, 'spin') == {} and nx.get_node_attributes(graph, 'field') == {} and \
                nx.get_edge_attributes(graph, 'coupling') == {}:
            # Initialize the spin, fields, and coupling on the graph
            self.graph = IsingGraph.initialize_graph(graph, coupling_strength = coupling_strength)
        else:
            # Assume the graph is properly initialized
            self.graph = graph

        self.temperature = temperature
        self.initializer = initializer

    @staticmethod
    def initialize_graph(graph, initializer = 'random', field_strength = 0.0, coupling_strength = 1.0):
        '''Takes an undirected graph and stores spin information on the nodes and coupling on the edges'''
        for node in graph.nodes:
            if initializer == 'random':
                graph.nodes[node]['spin'] = 1 if np.random.rand() > .5 else -1
            elif initializer == 'up':
                graph.nodes[node]['spin'] = 1
            elif initializer == 'down':
                graph.nodes[node]['spin'] = -1
            else:
                raise ValueError()

            graph.nodes[node]['field'] = field_strength

        for edge in graph.edges:
            graph.edges[edge]['coupling'] = -1 * coupling_strength

        return graph

    @staticmethod
    def visualize_graph(graph, pos = None):
        node_colors = list(nx.get_node_attributes(graph, 'spin').values())
        edge_colors = list(nx.get_edge_attributes(graph, 'coupling').values())
        pos = nx.nx_pydot.graphviz_layout(graph) if pos is not None else pos
        node_size = 300 * np.sqrt(25 / graph.number_of_nodes())
        nx.drawing.draw(graph, pos, node_color = node_colors, edge_color = edge_colors,
                        node_size = node_size, vmax = 1.0, vmin = -1.0)

    def get_energy_at_site(self, node):
        """
        Compute the energy at a given site in the lattice.
        """
        e = 0.0
        s1 = self.graph.nodes[node]['spin']

        # Get the neighboring spins of pos
        for neighbor in self.graph.neighbors(node):
            J = self.graph.edges[node, neighbor]['coupling']
            s2 = self.graph.nodes[neighbor]['spin']
            e += J * s1 * s2

        # Add contribution from field
        h = self.graph.nodes[node]['field']
        e += h * s1

        return e

    def metropolis_step(self):
        """
        Runs one step of the Metropolis-Hastings algorithm
        :return:
        """

        # Randomly select a site on the lattice
        node = np.random.choice(self.graph.nodes)

        # Calculate energy of the spin and energy if it is flipped
        energy = self.get_energy_at_site(node)

        energy_flipped = -1 * energy

        # Flip the spin if it is energetically favorable. If not, flip based on Boltzmann factor
        if energy_flipped <= 0:
            self.graph.nodes[node]['spin'] *= -1
        elif np.exp(-energy_flipped / self.temperature) > np.random.rand():
            self.graph.nodes[node]['spin'] *= -1

    def run(self, epochs, video = False, show_progress = False):

        iterator = tqdm(range(epochs)) if show_progress else range(epochs)

        if video:
            num_frames = 100
            FFMpegWriter = anim.writers['ffmpeg']
            writer = FFMpegWriter(fps = 10)

            # plt.ion()
            fig = plt.figure()

            with writer.saving(fig, "ising.mp4", 100):
                pos = nx.nx_pydot.graphviz_layout(self.graph)
                for epoch in iterator:
                    self.metropolis_step()
                    if epoch % (epochs // num_frames) == 0:
                        IsingGraph.visualize_graph(self.graph, pos = pos)
                        writer.grab_frame()
                        plt.clf()

            plt.close('all')

        else:
            for epoch in iterator:
                self.metropolis_step()


if __name__ == "__main__":
    graph = nx.convert_node_labels_to_integers(nx.generators.grid_2d_graph(20, 20))
    lattice = IsingGraph(graph)
    lattice.run(10000, video = True, show_progress = True)
