from evolutionaryAlgorithm import evolutionaryAlgorithm
from geneticProblem import geneticProblem
# from datetime import datetime, timedelta
import datetime

#jobshop scheduling
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import matplotlib.pyplot as plt


class JobShop(geneticProblem):
    def __init__(self, path, populationSize):
        self.data = list()
        self.population = []
        self.populationSize = populationSize
        self.operations = []
        self.machines =[]
        self.processingTime = []
        self.makespan = []

        with open(path, "r") as file:
            # Number of jobs and machines
            parameters = file.readline().strip().split()
            self.numJobs = int(parameters[0])
            self.numMachines = int(parameters[1])
            # Dataset: [(machine, processing time), ...] for each job
            for line in file:
                job_operations = []
                val = line.strip().split(' ')
                for i in range(0, len(val), 2):
                    job_operations.append((int(val[i]), int(val[i+1])))
                self.data.append(job_operations)

        #store each jobs machine sequence and processing time for operations
        for i in range(self.numJobs):
            job = []
            jtime = []
            for j in range(self.numMachines):
                job.append(self.data[i][j][0])
                jtime.append(self.data[i][j][1])
            self.machines.append(job)
            self.processingTime.append(jtime)
    
    def initializePopulation(self):
        population = []
        for i in range(self.populationSize):
            p = list(np.random.permutation(self.numMachines* self.numJobs) % self.numJobs)
            population.append(p)
        self.population = population
        return population
    


    def crossover(self, parent1, parent2):

        # Create a copy of parent1 to serve as the base for the child chromosome
        child = parent1[:]

        # Select two cut points randomly without replacement
        cutpoint = list(np.random.choice(len(parent1), 2, replace=False))
        cutpoint.sort()

        # Segment to be swapped from parent2
        segment_p2 = parent2[cutpoint[0]:cutpoint[1]]

        # Replace the segment in child with segment from parent2
        child[cutpoint[0]:cutpoint[1]] = segment_p2

        # To maintain the machine limit, we adjust the rest of the child chromosome
        # This involves ensuring no job appears more than numMachines times
        for job_id in range(self.numJobs):
            count_in_segment = segment_p2.count(job_id)
            total_count_in_child = child.count(job_id)
            over_count = total_count_in_child - self.numMachines

            # If more instances of the job exist than allowed, remove the excess
            if over_count > 0:
                for i in range(over_count):
                    # Find and remove the first occurrence outside the swapped segment
                    for idx in range(len(child)):
                        if child[idx] == job_id and not (cutpoint[0] <= idx < cutpoint[1]):
                            child[idx] = None  # Mark for removal
                            break

        # Fill the None values with missing jobs ensuring the job machine limit
        for job_id in range(self.numJobs):
            count_in_child = child.count(job_id)
            missing_count = self.numMachines - count_in_child
            none_positions = [idx for idx, val in enumerate(child) if val is None]

            for i in range(missing_count):
                child[none_positions[i]] = job_id

        return child


    def mutate(self, chromosome, mutationRate):
        mutation_prob = np.random.rand()
        if mutationRate >= mutation_prob:
            index = list(np.random.choice(self.numJobs, 2, replace=False))
            temp = chromosome[index[0]]
            chromosome[index[0]] = chromosome[index[1]]
            chromosome[index[1]] = temp
        return chromosome
    
    def fitnessFunction(self, population):
    
        makespans = []  # Store the makespan of each chromosome in the population
        for chromosome in population:
            # Initialize trackers for job completion times and machine availability times
            jobCompletionTimes = [0] * self.numJobs
            machineAvailabilityTimes = [0] * self.numMachines
            
            # Track the operation sequence for each job to ensure sequential operation processing
            jobOperationIndex = [0] * self.numJobs
            
            # Simulate the scheduling process based on the chromosome
            for gene in chromosome:
                jobID = gene
                operationIndex = jobOperationIndex[jobID]
                machineID, processingTime = self.data[jobID][operationIndex]
                
                # Determine the earliest start time for this operation
                earliestStartTime = max(jobCompletionTimes[jobID], machineAvailabilityTimes[machineID])
                completionTime = earliestStartTime + processingTime
                
                # Update job and machine trackers
                jobCompletionTimes[jobID] = completionTime
                machineAvailabilityTimes[machineID] = completionTime
                jobOperationIndex[jobID] += 1
            
            # The makespan is the maximum completion time across all jobs
            makespan = max(jobCompletionTimes)
            makespans.append(makespan)
        
        return makespans
       

    def plotGantt(self, chromosome, parent_scheme, survivor_scheme):
        m_keys = [j+1 for j in range(self.numMachines)]
        j_keys = [j for j in range(self.numJobs)]
        key_count = {key:0 for key in j_keys}
        j_count = {key:0 for key in j_keys}
        m_count = {key:0 for key in m_keys}
        j_record = {}
        for i in chromosome:
            gen_t = int(self.processingTime[i][key_count[i]])
            gen_m = int(self.machines[i][key_count[i]]) +1
            j_count[i] = j_count[i] + gen_t
            m_count[gen_m] = m_count[gen_m] + gen_t

            if m_count[gen_m] < j_count[i]:
                m_count[gen_m] = j_count[i]
            elif m_count[gen_m] > j_count[i]:
                j_count[i] = m_count[gen_m]

            start_time = str(datetime.timedelta(seconds = j_count[i] - self.processingTime[i][key_count[i]]))

            end_time = str(datetime.timedelta(seconds = j_count[i]))

            j_record[(i, gen_m)] = [start_time, end_time]

            key_count[i] = key_count[i] + 1
        
        df = []
        for m in m_keys:
            for j in j_keys:
                df.append(dict(Task='Machine %s'%(m), Start='2024-02-11 %s'%(str(j_record[(j,m)][0])), \
                                Finish='2024-02-11 %s'%(str(j_record[(j,m)][1])),Resource='Job %s'%(j+1)))
        
        df_ = pd.DataFrame(df)
        df_.Start = pd.to_datetime(df_['Start'])
        df_.Finish = pd.to_datetime(df_['Finish'])
        start = df_.Start.min()
        end = df_.Finish.max()

        df_.Start = df_.Start.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))
        df_.Finish = df_.Finish.apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))
        
        fig = px.timeline(df_, x_start="Start", x_end="Finish", y="Task", color="Resource", title='Job shop Schedule')
        fig.update_yaxes(categoryorder = 'total ascending')
        #save the plot
        fig.write_html(f"{parent_scheme}-{survivor_scheme}.html")
        fig.show()
    
    
       
path = "i1.txt"
populationSize = 1000

question = JobShop(path, populationSize)
isMax = False 
parentScheme = "Binary Tournament"
survivorScheme = "Random"
population, fitness = evolutionaryAlgorithm(question, isMax, parentScheme, survivorScheme)
best_index = fitness.index(max(fitness)) if isMax else fitness.index(min(fitness))
best_chromosome = population[best_index]
best_fitness_value = max(fitness) if isMax else min(fitness)
print(best_chromosome)
print(best_fitness_value)
question.plotGantt(best_chromosome,parentScheme, survivorScheme)





