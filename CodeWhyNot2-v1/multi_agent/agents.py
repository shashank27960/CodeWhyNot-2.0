import random

class GeneticAgent:
    """
    Genetic algorithm agent for evolving prompt variants.
    """
    def __init__(self, synonym_dict=None, population_size=6, generations=2, mutation_rate=0.3):
        self.synonym_dict = synonym_dict or {}
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def mutate(self, prompt):
        tokens = prompt.split()
        for i, token in enumerate(tokens):
            if token in self.synonym_dict and random.random() < self.mutation_rate:
                tokens[i] = random.choice(self.synonym_dict[token])
        return ' '.join(tokens)

    def crossover(self, prompt1, prompt2):
        t1 = prompt1.split()
        t2 = prompt2.split()
        if not t1 or not t2:
            return prompt1
        cut = random.randint(1, min(len(t1), len(t2)) - 1)
        return ' '.join(t1[:cut] + t2[cut:])

    def generate_candidates(self, prompt):
        # Initialize population
        population = [prompt]
        for _ in range(self.population_size - 1):
            population.append(self.mutate(prompt))
        # Evolve
        for _ in range(self.generations):
            # Crossover
            children = []
            for i in range(0, len(population) - 1, 2):
                children.append(self.crossover(population[i], population[i+1]))
            # Mutate
            children = [self.mutate(child) for child in children]
            population += children
            # Keep best unique prompts
            population = list(set(population))[:self.population_size]
        return population

class GradientAgent:
    """
    Gradient-based agent (stub, for LLMs with logprobs/gradients).
    """
    def generate_candidates(self, prompt):
        # Placeholder: return the original prompt for now
        return [prompt]

class RandomizedAgent:
    """
    Randomized agent using Gumbel noise and template edits.
    """
    def __init__(self, synonym_dict=None, p_perturb=0.2):
        self.synonym_dict = synonym_dict or {}
        self.p_perturb = p_perturb

    def generate_candidates(self, prompt):
        tokens = prompt.split()
        candidates = []
        for _ in range(4):
            new_tokens = [random.choice(self.synonym_dict[t]) if t in self.synonym_dict and random.random() < self.p_perturb else t for t in tokens]
            candidates.append(' '.join(new_tokens))
        return candidates 