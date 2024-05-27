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

target = plt.imread('scream.jpg')
img_shape = (25, 25)
target = resize(target, img_shape, preserve_range=True)

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

Individual.get_fitness = get_fitness
Population.get_entropy = phenotypic_entropy
Population.get_variance = phenotypic_variance

kwargs = {
    'shape': img_shape,
    'poly_range': [3, 5],
    'vertices_range': [3, 5]
}

evolve_kwargs = {'gens' : 2000,
                 'selection' : [tournament_selection, fps],
                 'selec_alg' : 1,
                 'tour_size': 0.1,
                 'mutation' : [polygon_mutation, pixel_mutation_random],
                 'mut_prob' : 0.04,
                 'mutation_alg_prob' : 1,
                 'pixel_mutation_same_color': True,
                 'mut_vertices_range' : [3,5],
                 'mut_poly_range' : [1,1],
                 'mut_pixel_range' : [img_shape[0]*img_shape[1]*0.005, img_shape[0]*img_shape[1]*0.01],
                 'crossover' : [blend_crossover, cut_crossover, pixel_crossover],
                 'xo_prob' : 0.8,
                 'xo_alg_prob' : [0.7, 0.25, 0.05],
                 'mirror_prob' : 0.05,
                 'elitism' : True,
                 'fitness_sharing' : False,
                 'fs_sigma' : 1,
                 'early_stopping' : 1500,
                 'verbose' : False}

#L = [[0.5, 0.45, 0.05],
     #[0.7, 0.25, 0.05],
     #[0.25, 0.7, 0.05],
     #[0.333, 0.333, 0.333],
     #[0.25, 0.05, 0.7],
     #[0.05, 0.25, 0.7],
     #[1, 0, 0],
     #[0, 1, 0],
     #[0, 0, 1] ]


pop = Population(size = 9, optim = 'min', **kwargs)
pop.grid(tuned_parameter = 'pop_size',
         param_range = [10, 50, 100],
         num_trials = 100,
         alpha = 0.05,
         save_dir = f"pop_size_final",
         **evolve_kwargs)

