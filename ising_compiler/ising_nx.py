import matplotlib as mpl
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm


class IsingGraph:

    def __init__(self, graph: nx.Graph, temperature = 0.5, initializer = 'random', coupling_strength = 1.0):

        assert type(graph) is nx.Graph

        self.graph = graph
        # Initialize graph attributes
        if nx.get_node_attributes(graph, 'spin') == {}:
            self.graph = IsingGraph.initialize_spins(self.graph, initializer = initializer)
        if nx.get_node_attributes(graph, 'field') == {}:
            self.graph = IsingGraph.initialize_fields(self.graph, field_strength = 0.0)
        if nx.get_edge_attributes(graph, 'coupling') == {}:
            self.graph = IsingGraph.initialize_couplings(self.graph, coupling_strength = coupling_strength)

        self.temperature = temperature
        self.initializer = initializer

    @staticmethod
    def initialize_spins(graph, initializer = 'random'):
        '''Takes an undirected graph and stores spin information on the nodes'''
        for node in graph.nodes:
            if initializer == 'random':
                graph.nodes[node]['spin'] = 1 if np.random.rand() > .5 else -1
            elif initializer == 'up':
                graph.nodes[node]['spin'] = 1
            elif initializer == 'down':
                graph.nodes[node]['spin'] = -1
            else:
                raise ValueError()

        return graph

    @staticmethod
    def initialize_fields(graph, field_strength = 0.0):
        '''Takes an undirected graph and stores field information on the nodes '''
        for node in graph.nodes:
            graph.nodes[node]['field'] = field_strength

        return graph

    @staticmethod
    def initialize_couplings(graph, coupling_strength = 1.0):
        '''Takes an undirected graph and stores coupling on the edges'''

        for edge in graph.edges:
            graph.edges[edge]['coupling'] = -1 * coupling_strength

        return graph

    @staticmethod
    def visualize_graph(graph, pos = None, fig = None, ax = None):

        if fig is None or ax is None:
            fig = plt.figure(figsize = (15, 10))
            ax = fig.add_subplot(1, 1, 1)

        if pos is None:
            pos = nx.nx_pydot.graphviz_layout(graph)

        node_colors = list(nx.get_node_attributes(graph, 'spin').values())
        edge_colors = list(nx.get_edge_attributes(graph, 'coupling').values())

        node_size = 4000 * np.sqrt(1 / graph.number_of_nodes())

        node_fields = list(nx.get_node_attributes(graph, 'field').values())
        field_sizes = node_size * (1 + 2 * np.sqrt(np.abs(node_fields)))

        # color axis
        cmap = plt.get_cmap('PiYG')
        cmap_fields = plt.get_cmap('bwr')

        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes('right', size = '2%', pad = '2%')
        mpl.colorbar.ColorbarBase(cax1, cmap = cmap, norm = mpl.colors.Normalize(-1, 1), ticks = [-1, +1])
        cax1.set_ylabel('Spin', rotation = 270, labelpad = -5, fontsize = 14)

        cax2 = divider.append_axes('right', size = '2%', pad = '4%')
        mpl.colorbar.ColorbarBase(cax2, cmap = cmap_fields, norm = mpl.colors.Normalize(-1, 1), ticks = [-1, +1])
        cax2.set_ylabel('Field strength, coupling', rotation = 270, labelpad = -5, fontsize = 14)

        node_labels = {n: n for n in graph.nodes() if not n.isdigit()}

        nx.draw_networkx_nodes(graph, pos, ax = ax, node_color = node_fields, vmax = 1.0, vmin = -1.0,
                               cmap = cmap_fields, node_size = field_sizes, alpha = .25)

        nx.drawing.draw(graph, pos, ax = ax, node_color = node_colors, edge_color = edge_colors,
                        node_size = node_size, vmax = 1.0, vmin = -1.0, width = 4,
                        cmap = cmap, edge_cmap = cmap_fields, labels = node_labels, font_size = 14,
                        font_color = "white")

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

    def run_metropolis(self, epochs, anneal_temperature_range = None, video = False, show_progress = False):

        iterator = tqdm(range(epochs)) if show_progress else range(epochs)
        if anneal_temperature_range:
            temperatures = np.geomspace(anneal_temperature_range[0], anneal_temperature_range[1], epochs)
        else:
            temperatures = [self.temperature] * epochs

        if video:
            num_frames = 100
            FFMpegWriter = anim.writers['ffmpeg']
            # writer = FFMpegWriter(fps = 10)

            pos = nx.nx_pydot.graphviz_layout(self.graph)

            # with writer.saving(fig, "ising.mp4", 100):
            for epoch in iterator:
                self.temperature = temperatures[epoch]
                self.metropolis_step()
                if epoch % (epochs // num_frames) == 0:
                    IsingGraph.visualize_graph(self.graph, pos = pos)
                    title = str(epoch).zfill(5)
                    plt.savefig(f"frames/{title}.png", dpi = 144)
                    plt.close()
                    # writer.grab_frame()
                    # plt.clf()

            plt.close('all')

        else:
            for epoch in iterator:
                self.temperature = temperatures[epoch]
                self.metropolis_step()


if __name__ == "__main__":
    graph = nx.convert_node_labels_to_integers(nx.generators.grid_2d_graph(20, 20))
    lattice = IsingGraph(graph)
    lattice.run_metropolis(10000, video = True, show_progress = True)
