import networkx as nx

class SCMModel:
    """
    Structural Causal Model for prompt interventions.
    Represents prompt nodes (NL tokens, concepts) and allows interventions (e.g., swap loop → recursion).
    """
    def __init__(self):
        # Initialize SCM graph (e.g., using NetworkX)
        pass

    def prompt_to_graph(self, prompt):
        """Parse prompt into a simple directed graph where each token is a node."""
        tokens = prompt.split()
        G = nx.DiGraph()
        for i, token in enumerate(tokens):
            G.add_node(i, label=token)
            if i > 0:
                G.add_edge(i-1, i)
        return G

    def graph_to_prompt(self, G):
        """Reconstruct prompt from graph node labels in order."""
        labels = [G.nodes[i]['label'] for i in sorted(G.nodes)]
        return ' '.join(labels)

    def intervene(self, prompt, intervention):
        """Apply an intervention to the prompt and return the modified prompt."""
        G = self.prompt_to_graph(prompt)
        if intervention == "Swap loop → recursion":
            for i in G.nodes:
                if G.nodes[i]['label'] == 'loop':
                    G.nodes[i]['label'] = 'recursion'
        elif intervention == "Add import":
            # Add 'import' at the start if not present
            if G.nodes[0]['label'] != 'import':
                mapping = {i: i+1 for i in G.nodes}
                G = nx.relabel_nodes(G, mapping)
                G.add_node(0, label='import')
                G.add_edge(0, 1)
        elif intervention == "Change variable name":
            # Change first variable-like token (simple heuristic)
            for i in G.nodes:
                if G.nodes[i]['label'].isidentifier() and G.nodes[i]['label'] not in ['for', 'if', 'def', 'return', 'import']:
                    G.nodes[i]['label'] = G.nodes[i]['label'] + '_new'
                    break
        # Add more interventions as needed
        return self.graph_to_prompt(G) 