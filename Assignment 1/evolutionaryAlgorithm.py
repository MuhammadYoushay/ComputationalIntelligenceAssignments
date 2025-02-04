import matplotlib.pyplot as plt
from selectionSchemes import SelectionSchemes

# EA Parameters
offspringCount = 50
numGenerations = 10000
mutationRate = 0.5
numIterations = 5
populationSize = 100

# This EA func iteratively evolves a population over multiple generations and iterations. During each generation,it evaluates the
# fitness of individuals, selects parents based on a specified parent selection scheme, generates offspring through 
# crossover and mutation, incorporates the offspring into the population, and selects survivors based on a 
# specified survivor selection scheme. The process is repeated for a predefined number of generations and iterations,
# and the function returns the final population and corresponding fitness values after the evolutionary process. 

def evolutionaryAlgorithm(question, isMax, parent_scheme, survivor_scheme):
    bestFitness, avgFitness = [0] * numGenerations, [0] * numGenerations
    population = question.initializePopulation()
    for iteration in range(numIterations):
        for generation in range(numGenerations):
            fitnessVals = question.fitnessFunction(population)
            maxFitness = max(fitnessVals) if isMax else min(fitnessVals)
            bestFitness[generation] += maxFitness
            avgFitness[generation] += sum(fitnessVals) / len(fitnessVals)
            parentSelection = SelectionSchemes(population, fitnessVals, 2 * offspringCount, isMax)
            parents = parentSelection.selectScheme(parent_scheme)
            offspring = [question.mutate(question.crossover(parents[i], parents[i + 1]), mutationRate) for i in range(offspringCount)]
            population.extend(offspring)
            fitnessVals.extend(question.fitnessFunction(offspring))
            eliteIndex = fitnessVals.index(maxFitness) if isMax else fitnessVals.index(min(fitnessVals))
            elite = population[eliteIndex]
            survivorSelection = SelectionSchemes(population, fitnessVals, populationSize, isMax)
            population = survivorSelection.selectScheme(survivor_scheme)
            if elite not in population:
                population[0] = elite
            print(f"iteration: {iteration}, generation: {generation}, best fitness: {maxFitness}, avg fitness: {sum(fitnessVals) / len(fitnessVals)}")
        print("Iteration complete!")

    avgBest = [fitness / numIterations for fitness in bestFitness]
    avgAvg = [fitness / numIterations for fitness in avgFitness]
    plotFunc(numGenerations, avgBest, avgAvg, parent_scheme, survivor_scheme)
    finalFVals = question.fitnessFunction(population)
    return population, finalFVals


def plotFunc(numGenerations, avgBest, avgAvg, parent_scheme, survivor_scheme):
    plt.plot(range(numGenerations), avgBest)
    plt.plot(range(numGenerations), avgAvg)
    plt.legend(["Average Best Fitness", "Average Average Fitness"])
    plt.title(f"Parent Selection: {parent_scheme} - Survivor Selection: {survivor_scheme}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()