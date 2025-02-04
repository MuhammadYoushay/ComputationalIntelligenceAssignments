# Computational Intelligence Assignments  
This repository contains my assignments for the **CS 451 - Computational Intelligence** course at **Habib University (Spring 2024)**. Each assignment explores various AI techniques, including **Evolutionary Algorithms, Swarm Intelligence, Reinforcement Learning, and Self-Organizing Maps (SOMs).**  

## Table of Contents  
- [Assignment 1: Evolutionary Algorithms](#assignment-1-evolutionary-algorithms)  
- [Assignment 2: Swarm Intelligence](#assignment-2-swarm-intelligence)  
- [Assignment 3: Reinforcement Learning and Self-Organizing Maps](#assignment-3-reinforcement-learning-and-self-organizing-maps)  
- [Setup Instructions](#setup-instructions)  
- [Results and Analysis](#results-and-analysis)  

---

## Assignment 1: Evolutionary Algorithms  

### Objective  
The goal of this assignment is to implement and analyze **Evolutionary Algorithms (EA)** to solve computationally hard problems, including:  

- **Traveling Salesman Problem (TSP)**  
- **Job-shop Scheduling Problem (JSSP)**  
- **Evolutionary Art (Human Image Evolution)**  

### Tasks & Implementation  
#### 1. Traveling Salesman Problem (TSP)  
- Implement EA to find an optimal path for visiting **194 cities in the Qatar dataset**.  
- **Dataset:** [TSP Problem Instances](http://www.math.uwaterloo.ca/tsp/world/countries.html).  
- Compare different selection schemes such as **Fitness Proportional Selection, Rank-based Selection, and Binary Tournament Selection**.  
- Visualize the evolution process using graphs.  

#### 2. Job-shop Scheduling Problem (JSSP)  
- Implement EA to optimize scheduling of tasks on multiple machines.  
- Use instances **abz5-abz7** from [JSSP datasets](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt).  
- Ensure **modular and object-oriented design** for reusability.  

#### 3. Evolutionary Art (Human Image Evolution)  
- Modify EA to **evolve an image using polygons**.  
- Experiment with **different images apart from Mona Lisa**.  
- Optimize **mutation and crossover techniques** for efficient image evolution.  

### Evaluation Metrics  
- **Problem formulation** (15%)  
- **EA implementation correctness & structuring** (40%)  
- **Parameter tuning & optimal solutions** (10%)  
- **Performance visualization through graphs** (15%)  
- **Analysis and findings** (20%)  

---

## Assignment 2: Swarm Intelligence  

### Objective  
This assignment involves implementing **Swarm Intelligence-based algorithms** to solve optimization problems and develop a **visualization of swarm behavior**.  

### Tasks & Implementation  
#### 1. Graph Coloring using Ant Colony Optimization (ACO)  
- Implement ACO to solve the **graph vertex-coloring problem**.  
- Ensure that **adjacent vertices do not share the same color** while minimizing total colors used.  
- Test on datasets **queen11_11.col and le450-15b.col** from [Graph Coloring Instances](https://mat.tepper.cmu.edu/COLOR/instances.html).  
- Tune parameters (**α, β, γ, number of ants**) to optimize results.  
- Plot graphs:  
  - **Iteration vs Best Fitness**  
  - **Iteration vs Average Fitness**  

#### 2. Swarm Visualization (Simulation)  
- Develop a **Processing-based simulation** for any of the following:  
  - **Particle Swarm Optimization**  
  - **Particle Systems**  
  - **Ant Clustering**  
- Allow **parameter manipulation** and **real-time visualization** of swarm behavior.  

### Evaluation Metrics  
- **Problem formulation & correct ACO implementation** (40%)  
- **Fine-tuning & optimization** (15%)  
- **Visualization of swarm behavior** (50%)  
- **Interactivity (parameter manipulation & real-time effects)** (25%)  

---

## Assignment 3: Reinforcement Learning and Self-Organizing Maps  

### Objective  
This assignment focuses on **Self-Organizing Maps (SOMs) for clustering** and **Reinforcement Learning (RL) for decision-making**.  

### Tasks & Implementation  
#### 1. Clustering COVID-19 Data using SOM  
- Apply **Self-Organizing Maps (SOM)** to cluster country-wise COVID-19 data (**Confirmed, Recovered, Expired cases**).  
- Map data fields to **RGB values for color-coded visualization**.  
- Use **Sum of Squared Distance** for Best Matching Unit (BMU).  
- Implement **exponential decay for learning rate and neighborhood radius**.  
- Further enhance visualization by **mapping clusters onto a world map**.  
- **Dataset:** COVID-19 case data (Jan 2021).  

#### 2. Reinforcement Learning in FrozenLake  
- Implement **Value Iteration** to train an agent in the **FrozenLake environment** (OpenAI Gym).  
- The agent must **navigate safely from start (S) to goal (G)** without falling into holes (H).  
- Test on different grid sizes (e.g., **4x6, 8x8 grids**).  
- Evaluate **learned policies** and compare performance across different configurations.  

### Evaluation Metrics  

#### SOM Clustering (30 points)  
- **SOM process & convergence** (35%)  
- **Radius & learning rate decay** (10%)  
- **Color-coded visualization (SOM grid + World Map)** (30%)  

#### Reinforcement Learning (20 points)  
- **Correct implementation of Value Iteration** (35%)  
- **Value update and stopping criteria** (20%)  
- **Effective action selection and execution** (35%)  
- **Performance on multiple grid sizes** (10%)  

---

Feel free to explore, contribute, or provide feedback! 🚀
