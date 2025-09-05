import numpy as np
import random
import networkx as nx

class GumbelCounterfactualGenerator:
    """
    Generates counterfactual prompts using SCM graph and optional Gumbel-based token-level noise sampling.
    """
    def __init__(self, p_perturb=0.2, seed=None):
        self.p_perturb = p_perturb
        self.synonyms = {
            'loop': ['iteration', 'cycle'],
            'recursion': ['recursive', 'self-call'],
            'function': ['method', 'procedure'],
            'variable': ['var', 'identifier'],
            'calculate': ['compute', 'determine'],
            'sum': ['total', 'add'],
            'factorial': ['product', 'factor'],
        }
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def generate(self, scm_graph, intervention=None, use_gumbel=False):
        """
        Generate a counterfactual prompt from the SCM graph.
        Optionally apply an intervention and Gumbel noise.
        """
        G = scm_graph.copy()
        # Apply intervention at node level
        if intervention == "Swap loop → recursion":
            for i in G.nodes:
                if G.nodes[i]['label'] == 'loop':
                    G.nodes[i]['label'] = 'recursion'
        elif intervention == "Add import":
            if 0 not in G.nodes or G.nodes[0]['label'] != 'import':
                mapping = {i: i+1 for i in G.nodes}
                G = nx.relabel_nodes(G, mapping)
                G.add_node(0, label='import')
                G.add_edge(0, 1)
        elif intervention == "Change variable name":
            for i in G.nodes:
                if G.nodes[i]['label'].isidentifier() and G.nodes[i]['label'] not in ['for', 'if', 'def', 'return', 'import']:
                    G.nodes[i]['label'] = G.nodes[i]['label'] + '_new'
                    break
        # Optionally apply Gumbel noise
        if use_gumbel:
            for i in G.nodes:
                label = G.nodes[i]['label']
                if label in self.synonyms and random.random() < self.p_perturb:
                    G.nodes[i]['label'] = random.choice(self.synonyms[label])
        # Reconstruct prompt from graph
        labels = [G.nodes[i]['label'] for i in sorted(G.nodes)]
        return ' '.join(labels) 