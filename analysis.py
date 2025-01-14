import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import configparser
import json
from collections import deque
from buildgraph import *

config = configparser.RawConfigParser()
config.read('params.ini')
n_value = int(config['params']['n_value'])
alpha_value = float(config['params']['alpha_value'])
beta_value = float(config['params']['beta_value'])
balance_value = float(config['params']['balance_value'])
d_value = int(config['params']['d_value'])
lambda_decay = float(config['params']['lambda_decay'])
random_seed = int(config['params']['random_seed'])

class GraphAnalysis:

    # initialize a graph object parameters created from the buildgraph class
    def __init__(self, graph, positions, weights, lambda_decay):
        self.graph = graph
        self.positions = positions
        self.weights = weights
        self.lambda_decay = lambda_decay

    def exponential_info_spread(self, start_node):
        reached = {}
        queue = deque([(start_node, 0)])

        # traversal algorithm, BFS discounted by depth
        while queue:
            curr_node, curr_depth = queue.popleft()

            if curr_node not in reached or reached[curr_node] > curr_depth:
                reached[curr_node] = curr_depth

                for neighbor in self.graph.neighbors(curr_node):
                    # assign each connection an arbitrary threshold (i.e. influence)
                    # and compare to an exponentially decaying level of influence,
                    # based on distance from source
                    influence = math.exp(-self.lambda_decay * (curr_depth + 1))
                    viewing_threshold = random.gauss(0.5, 0.1)
                    if influence > viewing_threshold:
                        queue.append((neighbor, curr_depth + 1))

        return reached

    def write_spread_analytics(self, start_node, reached):
        levels = {}
        for node, depth in reached.items():
            if depth not in levels:
                levels[depth] = []
            levels[depth].append(node)

        with open('spread_analytics.txt', 'w') as f:
            f.write(f"Spread seeded at {random_seed} starting from node {start_node}\n")
            for level, nodes in sorted(levels.items()):
                nodes_str = ", ".join(map(str, nodes))
                f.write(f"Depth {level}: Includes node [{nodes_str}]\n")

    def draw_spread(self, start_node):

        # Perform an information spread simulation from one node
        reached = self.exponential_info_spread(start_node)

        # Create the original graph for comparison
        sizes = [50 * self.weights[node][0] for node in self.graph.nodes()]
        nx.draw(self.graph, pos=self.positions, node_size=sizes, edgecolors="black", node_color='lightblue')

        # Highlight all reached nodes
        colors = []
        for node in self.graph.nodes():
            if node == start_node:
                colors.append('red')
            elif node in reached:
                colors.append('orange')
            else:
                colors.append('lightblue')

        labels = {start_node: "$V_0$"}

        nx.draw(self.graph, pos=self.positions, node_size=sizes, node_color=colors, edgecolors="black", with_labels=False)
        nx.draw_networkx_labels(self.graph, pos=self.positions, labels=labels, font_size=22, font_color="black")

        # Add an arrow to highlight the starting node
        pos_start_node = self.positions[start_node]
        plt.annotate("Start", xy=pos_start_node, xycoords='data',
                     xytext=(20, 20), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='red'),
                     fontsize=12, color='red', weight='bold')

        plt.title("Sample Spread Drawn from Arbitrary Point")
        plt.show()

        # Write the spread analytics to a text file
        self.write_spread_analytics(start_node, reached)

def print_original_graph(graph, positions, weights):
    sizes = [50 * weights[node][0] for node in graph.nodes()]
    nx.draw(graph, pos=positions, node_size=sizes, edgecolors="black", node_color='lightblue')
    plt.title("Original Graph")
    plt.show()

def main():

    # control the random seed
    random.seed(200)
    np.random.seed(200)

    # Build the graph from the buildgraph class
    edgeDict, posDict, nodeWeightDict = buildGraph(n_value, alpha_value, beta_value, balance_value, d_value)
    G = nx.from_dict_of_lists(edgeDict, create_using=nx.DiGraph)

    # Print the original graph
    print_original_graph(G, posDict, nodeWeightDict)

    # Show the spread on the same model for comparison
    spread_model = GraphAnalysis(G, posDict, nodeWeightDict, lambda_decay)
    start_node = random.choice(list(G.nodes))
    spread_model.draw_spread(start_node)

if __name__ == "__main__":
    main()