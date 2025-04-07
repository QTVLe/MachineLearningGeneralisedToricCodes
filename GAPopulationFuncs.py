import numpy as np
from SageFunctions_ import list2vertices, vertices2list
from random import choice

# GA finctions
    
# Custom crossover function to not allow repetition of vertices
def crossover_func(parents, offspring_size, ga_instance): # offspring_size = the offspring size, number of genes
    offspring = np.zeros(offspring_size, dtype=int)
    counter = 0
    while counter < offspring_size[0]:
        # choose two random parents from parents
        idx1 = np.random.randint(low = 0, high = parents.shape[0])
        idx2 = np.random.randint(low = 0, high = parents.shape[0])

        # convert to  vertices (sorted, lower chance to get repetitions of vertices?)
        parent1 = list2vertices(parents[idx1, :])
        parent2 = list2vertices(parents[idx2, :])

        random_split_point = np.random.choice( range(offspring_size[1] // 2) ) # split while preserving vertices (pairs of coords)

        parent1[random_split_point:] = parent2[random_split_point:]

        if len( set(parent1) ) < (offspring_size[1] // 2): # check for repeating vertices
            continue # try again

        offspring[counter] = vertices2list(parent1)
        counter += 1

    return offspring

# Custom mutation function to not allow repetition of vertices
def mutation_func(sageClass_instance, offspring, ga_instance):
    mutation_chance = ga_instance.mutation_percent_genes / 100.
    offspring_size = offspring.shape[0]

    allvertices_set = set(sageClass_instance.allvertices)

    for chromosome_idx in range(offspring_size):
        vertices = list2vertices(offspring[chromosome_idx])
        vertices_mutated = set(vertices)
        # construct mutated gene sequence and check if it is valid
        for vertex in vertices:
            if np.random.random() < mutation_chance:
                difference = list(allvertices_set.difference(vertices_mutated))

                new_vertex = choice(difference)

                vertices_mutated.remove(vertex)
                vertices_mutated.add(new_vertex)
        offspring[chromosome_idx] = vertices2list(vertices_mutated)

    return offspring 

# Custom function to generate initial population without repetition of vertices
def generateInitialPopulation(SF, sol_per_pop, num_genes):
    '''Needs SageFunctions instance as a furst argument to work properly'''
    initial_population = SF.generate_sets_of_vertices(sol_per_pop, num_genes//2)
    population = np.array(initial_population, dtype=np.float32).reshape((sol_per_pop, num_genes))
    return population

def population2populationVertices(population):
    '''Function to convert population with Lists to Polulation with Vertices, should receive "ga_instance.population"'''
    populationVertices = []
    
    for individual in population:
        individual_vertices = list2vertices(individual)
        populationVertices.append( individual_vertices )
    
    return populationVertices

def eliminate_repeated_solutions(population):
    '''Function to eliminate repeated solution from population, should receive populationVertices as an input'''
    reduced_population = set()

    for individual in population:
        reduced_population.add( tuple(individual) )
    
    return list(reduced_population)

def reduce_population(SF, population, dimension):
    '''Function to eliminate repeated solution from population, should receive populationVertices as an input'''
    reduced_population = set()

    for individual in population:
        toric_Generator = SF.vertices2Generator(individual)
        dim = toric_Generator.shape[0]
        if dim == dimension:
            reduced_population.add( tuple(individual) )
    
    return list(reduced_population)

def vertices2MagmaStr(vertices):
    vertices_strings = []

    for vertex in vertices:
        vertices_strings.append( f"<{vertex[0]}, {vertex[1]}>" )

    return "[" + ", ".join(vertices_strings) + "]"

def populationVertices2MagmaStr(population):
    '''Converts populationVertices to Magma string'''
    population = eliminate_repeated_solutions(population)
    vertices_strings = []

    for individual in population:
        vertices_strings.append(vertices2MagmaStr(individual))

    return "[" + ", ".join(vertices_strings) + "]"

