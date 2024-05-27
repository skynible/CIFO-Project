import time
import random
import numpy as np
from charles_p import Individual, Population
from selection_p import fps, tournament_selection
from mutation_p import polygon_mutation, pixel_mutation_random
from xo_p import blend_crossover, cut_crossover, pixel_crossover
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from operator import attrgetter
from skimage.transform import resize

##########################################################
target = plt.imread('scream.jpg')
img_shape = (25, 25)
target = resize(target, img_shape, preserve_range=True)
##########################################################

################################### METRICS ##########################################
def phenotypic_entropy(self):

    """Calculates the phenotypic entropy of a population.

            Args:
                population: population of Individuals

            Returns:
                entropy: phenotypic_entropy
            """

    entropy = 0
    for individual in self:
        entropy += -individual.fitness*np.log(individual.fitness)
    return entropy

def phenotypic_variance(self):
    """Calculates the phenotypic variance of a population.

            Args:
                population: population of Individuals

            Returns:
                variance: phenotypic_variance
            """

    variance = 0
    n = len(self)
    fitness_mean = np.mean([individual.fitness for individual in self])
    for individual in self:
        variance += (1/(n-1))*((individual.fitness-fitness_mean)**2)
    return variance

######################################## DISTANCES ##############################################################
def delta_e(color1, color2):

    """Calculates the delta-e distance between two colors.

            Args:
                color1: array of rgb values
                color2: array of rgb values

            Returns:
                delta_e: float
            """

    diff_vec = color1 - color2
    red_avg = (color1[0] + color2[0])/2

    if red_avg<128:
        delta_e = np.sqrt((2*(diff_vec[0]**2))+(4*(diff_vec[1]**2))+(3*(diff_vec[2]**2)))

    else:
        delta_e = np.sqrt((3*(diff_vec[0]**2))+(4*(diff_vec[1]**2))+(2*(diff_vec[2]**2)))

    return delta_e

def get_fitness(self, target):

    delta_e_sum = 0
    for row in range(self.representation.shape[0]):
        for pixel in range(self.representation.shape[1]):
            delta_e_sum += delta_e(self.representation[row][pixel], target[row][pixel])

    return delta_e_sum

####################################### COMPILATION #############################################################

# monkey patch functions
Individual.get_fitness = get_fitness
Population.get_entropy = phenotypic_entropy
Population.get_variance = phenotypic_variance


kwargs = {
    'shape': img_shape, # Individual shape
    'poly_range': [2, 3], # possible number of polygons when starting the individual
    'vertices_range': [3, 5] # possible number of vertices  of polygons
}

# we did not check if the edges of the same polygon do not intercept

pop_size = 9

pop = Population(size = pop_size, optim = 'min', **kwargs)
history, best_individual = pop.evolve(gens = 50000,
                                      selection = [tournament_selection, fps],
                                      selec_alg = 1,
                                      tour_size = 2/9,
                                      mutation = [polygon_mutation, pixel_mutation_random],
                                      mut_prob = 0.01,
                                      mutation_alg_prob = 0.7,
                                      pixel_mutation_same_color = True,
                                      mut_vertices_range = [3,5],
                                      mut_poly_range = [1,1],
                                      mut_pixel_range = [img_shape[0]*img_shape[1]*0.005, img_shape[0]*img_shape[1]*0.01],
                                      crossover = [blend_crossover, cut_crossover, pixel_crossover],
                                      xo_prob = 0.6,
                                      xo_alg_prob = [0.7, 0.25, 0.05],
                                      mirror_prob = 0.1,
                                      elitism = True,
                                      fitness_sharing = False,
                                      fs_sigma = 1,
                                      early_stopping = 1500,
                                      verbose = True)


######################################## HISTORY ##########################################

print(f"best individual fitness: {best_individual.fitness}")

plt.imshow(target.astype(np.uint8))
plt.title('Target')
plt.show()

plt.imshow(best_individual.representation.astype(np.uint8))
plt.title('Best_result')
plt.show()

plt.minorticks_on()
plt.grid(which="minor",linestyle=":",linewidth=0.75)
plt.grid()
plt.scatter(history[0], history[1])
plt.title('Fitness History')
plt.ylabel("Best Fitness")
plt.xlabel("Generation")
plt.show()

plt.minorticks_on()
plt.grid(which="minor",linestyle=":",linewidth=0.75)
plt.grid()
plt.scatter(history[0], history[2])
plt.title('Entropy History')
plt.ylabel("Phenotypic Entropy")
plt.xlabel("Generation")
plt.show()

plt.minorticks_on()
plt.grid(which="minor",linestyle=":",linewidth=0.75)
plt.grid()
plt.scatter(history[0], history[3])
plt.title('Variance History')
plt.ylabel("Phenotypic Variance")
plt.xlabel("Generation")
plt.show()

