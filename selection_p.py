from random import uniform
import numpy as np


def fps(population):
    """Fitness proportionate selection implementation.

    Args:
        population (Population): The population we want to select from.

    Returns:
        parent1, parent2: selected parents (Individuals).
    """

    accumulated_prob = []
    if population.optim == "min":
        inverse_total_fitness = sum([1/i.fitness for i in population])

        prob = 0
        for individual in population:
            prob += ( (1 / individual.fitness) / inverse_total_fitness )
            accumulated_prob.append(prob)
        accumulated_prob[-1] = 1

        # Select 2 parents with random draws. The index in accumulated_prob identifies
        # the individual in the population. We also need to take into account the possibility
        # of selecting the same individual twice with same_parent bool.

        same_parent = True
        while same_parent == True:
            found_parent1 = False
            found_parent2 = False
            draw_parent1 = np.random.uniform()
            draw_parent2 = np.random.uniform()
            for index in range(len(accumulated_prob)):
                if draw_parent1 < accumulated_prob[index] and found_parent1 == False:
                    parent1_index = index
                    found_parent1 = True
                if draw_parent2 < accumulated_prob[index] and found_parent2 == False:
                    parent2_index = index
                    found_parent2 = True
            if parent1_index != parent2_index:
                same_parent = False
                parent1 = population[parent1_index]
                parent2 = population[parent2_index]

        return parent1, parent2


    elif population.optim == "max":
        raise NotImplementedError
    else:
        raise Exception(f"Optimization not specified (max/min)")


def tournament_selection(population, tournament_size):
    """Fitness proportionate selection implementation.

        Args:
            population (Population): The population we want to select from.
            tournament_size: size of the tournament

        Returns:
            parent1, parent2: selected parents (Individuals).
        """

    same_parent = True

    while same_parent:

        parent_list = []

        for i in range(2):
            best_fitness = 1e30
            best_ind = 0
            for _ in range(tournament_size):
                draw = np.random.randint(0,len(population))
                if best_fitness > population[draw].fitness:
                    best_ind = draw
            parent_list.append(population[best_ind])

        if parent_list[0].fitness != parent_list[1].fitness:
            same_parent = False

    return parent_list[0], parent_list[1]



