import networkx as nx
import json
import csv
from typing import Optional, List

class SCMEngine:
    def __init__(self, template_file: Optional[str] = None):
        self.graph = nx.DiGraph()
        if template_file:
            self.load_templates(template_file)
        else:
            # Default: simple causal relations
            self.graph.add_edge("loop", "recursion")
            self.graph.add_edge("for", "map")
            self.graph.add_edge("list comprehension", "generator expression")
            self.graph.add_edge("import math", "math.sqrt")

    def load_templates(self, path: str):
        if path.endswith('.json'):
            with open(path, 'r') as f:
                data = json.load(f)
            for edge in data.get("edges", []):
                self.graph.add_edge(edge["from"], edge["to"])
        elif path.endswith('.csv'):
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.graph.add_edge(row["from"], row["to"])

    def intervene(self, prompt: str, concept: str, new_value: str) -> str:
        # Replace concept in prompt with new_value (concept-level intervention)
        return prompt.replace(concept, new_value)

    def get_concepts(self) -> List[str]:
        return list(self.graph.nodes)

    def get_possible_interventions(self, concept: str) -> List[str]:
        # Return possible interventions for a concept (out-edges)
        return [t for t in self.graph.successors(concept)]

    def generate_counterfactual_prompt(self, prompt: str, concept: str, new_value: str) -> str:
        # Tokenize prompt, replace concept token, reconstruct prompt
        tokens = prompt.split()  # Simple whitespace tokenizer; replace with LLM tokenizer if needed
        new_tokens = [new_value if t == concept else t for t in tokens]
        return ' '.join(new_tokens) 