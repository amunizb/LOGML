#QUESTIONS
#1 Does this algorith work best for symmetrical or non-symmetrical objective functions?
# (I am particularly concerned about the slicing step--doesn't make much biological sense to me?)
# If symmetrical, then getting any subcollection of bits from mum and dad should be the same
# If non-symmetrical, then some variables affect objective function more. So...?

import numpy as np
import matplotlib.pyplot as plt
def objective(bits):
    return -sum(bits) 

def choose_parents(pop, k):
    n_pop = len(pop)
    parents=[]
    for _ in range(2):
        # Initialise random generator
        rng = np.random.default_rng()
        # Select k individuals randomly from the population
        candidates = rng.integers(0, n_pop, k)
        # Get the best individual from the group
        best = np.argmin([objective(pop[i]) for i in candidates])
        parents.append(pop[candidates[best]])
    return parents

def crossover(parents):
    child = parents[0][0:slice_point] + parents[1][slice_point:]
    return child


def genetic_algorithm():

    # Keep track of generational best and average score
    best_score =[]
    average_score=[]

    ### CREATE RANDOM POPULATION ###

    # Initialise random generator
    rng = np.random.default_rng()
    #Create population of size n_pop, each individual with size n_bits 
    pop = [rng.integers(0, 2, n_bits).tolist() for _ in range(n_pop)]

    # Initialise lists 
    best_score.append(min([objective(pop[i]) for i in range(n_pop)]))
    average_score.append(sum([objective(pop[i]) for i in range(n_pop)])/n_pop)
    #print(f"The initial population has maximum objective function of {min([objective(pop[i]) for i in range(n_pop)])}")

    #Iterate over n_iter generatiorns
    for iter in range(n_iter):
        new_pop = []
        for _ in range(n_pop):
            # Select parents and create child
            parents = choose_parents(pop, k)

            # Crossover parents in 'most' cases
            if rng.random() < crossover_rate:
                child = crossover(parents)
            # Copy best parent in 'few' cases
            else:
                best_parent = np.argmin([objective(parents[i]) for i in range(2)])
                child = parents[best_parent]

            # Mutate each bit with probability 1/n_bit
            for bit in range(n_bits):
                if rng.random() < 1/n_bits:
                    child[bit] = 1 - child[bit]
            new_pop.append(child)

        # Update current generation    
        pop = new_pop

        # Update lists
        best_score.append(min([objective(pop[i]) for i in range(n_pop)]))
        average_score.append(sum([objective(pop[i]) for i in range(n_pop)])/n_pop)
        #print(f"The population after {iter+1} mutations has a minimum objective function of {min([objective(pop[i]) for i in range(n_pop)])}")


    #Plot generational evolution
    fig, ax = plt.subplots()
    ax.plot(best_score, label="Best score")
    ax.plot(average_score, label="Average score")
    ax.plot(-n_bits * np.ones(n_iter), linestyle='--')
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective function')
    plt.show()

    best_idx = np.argmin([objective(x) for x in pop])
    return pop[best_idx], best_score[-1]

#Hyperparameters
n_pop = 1000 #Population size
n_bits = 20 #Number of bits for each individual
n_iter = 50 #Number of generations
slice_point = n_bits // 2 #Slice point for mutation
k = 3 #Tournament size for parent selection
crossover_rate = 0.9 #Crossover rate

best, best_score = genetic_algorithm()
print(f"The best individual is \n {best} \n with a score of {best_score}")