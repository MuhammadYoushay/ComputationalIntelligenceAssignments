import tsplib95
from random import randint, uniform
from geneticProblem import geneticProblem
from evolutionaryAlgorithm import evolutionaryAlgorithm
from evolutionaryAlgorithm import populationSize

class TSPProblem(geneticProblem):
    def __init__(self, path, populationSize):
      self.tspData = tsplib95.load(path)
      self.cities = len(list(self.tspData.get_nodes()))
      super().__init__(populationSize)
    
    # initialize a population with random paths covering all cities
    def initializePopulation(self):
        population = []
        for _ in range(self.populationSize):
            path = []
            while len(path) < self.cities:
                node = randint(1, self.cities)
                if node not in path:
                    path.append(node)
            population.append(path)
        return population

    # described alaborately in the report
    def fitnessFunction(self, chromosomes):
        fitnessScores = [None] * len(chromosomes)
        for i, path in enumerate(chromosomes):
            totalDistance = 0
            for j in range(1, len(path)):
                start, end = path[j - 1], path[j]
                totalDistance += self.tspData.get_weight(start, end)
            fitnessScores[i] = totalDistance
        return fitnessScores

    # Creates a offspring. Random material from parent 1, rest from parent 2
    def crossover(self, parent1, parent2):
        offspring = [None] * len(parent1)
        mid = len(parent1) // 2
        start = randint(1, mid - 2)
        end = start + mid
        offspring[start:end] = parent1[start:end]
        indexB = indexOffspring = end
        while None in offspring:
            if parent2[indexB] not in offspring:
                offspring[indexOffspring] = parent2[indexB]
                indexOffspring = (indexOffspring + 1) % len(parent2)
            indexB = (indexB + 1) % len(parent2)
        return offspring

    # Mutation - by swapping
    def mutate(self, chromosome, mutationRate):
        if uniform(0, 1) < mutationRate:
            pos1 = randint(1, len(chromosome) - 2)
            pos2 = pos1
            while pos1 == pos2:
                pos2 = randint(1, len(chromosome) - 2)
            chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]
        return chromosome

def main():
    tspFile = 'qa194.tsp'
    question = TSPProblem(tspFile, populationSize)
    isMax = False 
    pop, fitness = evolutionaryAlgorithm(question, isMax, "RBS", "Binary Tournament")
    best_index = fitness.index(max(fitness)) if isMax else fitness.index(min(fitness))
    bestChromosome = pop[best_index]
    bestFitness = max(fitness) if isMax else min(fitness)
    print('The best chromosome is ---> {}'.format(bestChromosome))
    print('The best fitness value is ---> {}'.format(bestFitness))

if __name__ == "__main__":
    main()
