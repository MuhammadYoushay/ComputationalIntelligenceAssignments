import numpy as np
import random 
class SelectionSchemes:
    def __init__(self, population, fitness_vals, population_size, is_max):
        self.population = population
        self.fitness_vals = np.array(fitness_vals, dtype=float)
        self.population_size = population_size
        self.is_max = is_max
    # Helper Func    
    def __fitness_proportional_values(self):
        if self.is_max:
            fitness_vals_cumulative = np.cumsum(self.fitness_vals / self.fitness_vals.sum())
        else:
            adjusted_fitness_vals = 1 / (self.fitness_vals + 1)
            fitness_vals_cumulative = np.cumsum(adjusted_fitness_vals / adjusted_fitness_vals.sum())
        return fitness_vals_cumulative
    # Method that performs fitness-proportional selection by creating a new population using the cumulative fitness values.
    # Individuals are chosen based on their fitness proportion.
    def fitness_proportional(self):
        fitness_vals_cumulative = self.__fitness_proportional_values()
        new_population = [self.population[np.searchsorted(fitness_vals_cumulative, random.random())] for _ in range(self.population_size)]
        return new_population
    # Method that performs random selection by randomly choosing individuals from the current population to form a new population.
    def random_scheme(self):
        return random.choices(self.population, k=self.population_size)
    # Method that performs binary tournament selection. For each position in the new population, two individuals
    # are randomly selected, and the one with higher (or lower, depending on the goal) fitness is chosen as the winner.
    def binary_tournament(self):
        new_population = []
        for _ in range(self.population_size):
            competitors = random.sample(list(enumerate(self.fitness_vals)), 2)
            winner = max(competitors, key=lambda c: c[1] if self.is_max else -c[1])
            new_population.append(self.population[winner[0]])
        return new_population
    # Method that performs truncation selection by selecting the top individuals in the current population based on their
    #fitness values.
    def truncation(self):
        sorted_indices = np.argsort(-self.fitness_vals if self.is_max else self.fitness_vals)
        return [self.population[index] for index in sorted_indices[:self.population_size]]
    # Method that performs rank-based selection. Individuals are assigned ranks based on their fitness values, 
    # and a new population is created by choosing individuals with probabilities proportional to their ranks.
    def rank_based(self):
        ranks = np.argsort(np.argsort(-self.fitness_vals if self.is_max else self.fitness_vals))
        scaled_ranks = ranks / ranks.sum()
        ranks_cumulative = np.cumsum(scaled_ranks)
        new_population = [self.population[np.searchsorted(ranks_cumulative, random.random())] for _ in range(self.population_size)]
        return new_population
    def selectScheme(self, selection_method):
        methods = {
            "Random": self.random_scheme,
            "Binary Tournament": self.binary_tournament,
            "FPS": self.fitness_proportional,
            "Truncation": self.truncation,
            "RBS": self.rank_based
        }
        if selection_method in methods:
            return methods[selection_method]()
        else:
            raise ValueError(f"Selection method {selection_method} not found!")