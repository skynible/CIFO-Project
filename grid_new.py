import time
import random
import numpy as np
from charles_p import Individual, Population
from selection_p import fps, tournament_selection
from mutation_p import polygon_mutation, pixel_mutation_random
from xo_p import blend_crossover, cut_crossover, pixel_crossover
from metrics_p import delta_e, get_fitness, phenotypic_variance
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from operator import attrgetter
from skimage.transform import resize

target = plt.imread('scream.jpg')
img_shape = (25, 25)
target = resize(target, img_shape, preserve_range=True)


Individual.get_fitness = get_fitness
Population.get_variance = phenotypic_variance

kwargs = {
    'shape': img_shape,
    'poly_range': [3, 5],
    'vertices_range': [3, 5]
}

evolve_kwargs = {'gens' : 2000,
                 'selection' : [tournament_selection, fps],
                 'selec_alg' : 0,
                 'tour_size': 0.2,
                 'mutation' : [polygon_mutation, pixel_mutation_random],
                 'mut_prob' : 0.07,
                 'mutation_alg_prob' : 0.9,
                 'pixel_mutation_same_color': True,
                 'mut_vertices_range' : [3,5],
                 'mut_poly_range' : [1,1],
                 'mut_pixel_range' : [img_shape[0]*img_shape[1]*0.005, img_shape[0]*img_shape[1]*0.01],
                 'crossover' : [blend_crossover, cut_crossover, pixel_crossover],
                 'xo_prob' : 0.6,
                 'xo_alg_prob' : [0.25, 0.7, 0.05],
                 'mirror_prob' : 0.1,
                 'elitism' : True,
                 'fitness_sharing' : False,
                 'fs_sigma' : 0.7,
                 'mutation_size' : img_shape[0],
                 'early_stopping' : 1500,
                 'verbose' : False}




pop = Population(size = 10, optim = 'min', **kwargs)
pop.grid(tuned_parameter = 'mutation_size', #pop_size is recognised as a tunable parameter
         param_range = [round(img_shape[0]*0.3),
                        round(img_shape[0]*0.4),
                        round(img_shape[0]*0.5),
                        round(img_shape[0]*0.6),
                        round(img_shape[0]*0.7),
                        round(img_shape[0]*0.8),
                        round(img_shape[0]*0.9),
                        round(img_shape[0])],

         num_trials = 100,
         alpha = 0.05,
         save_dir = f"mutation_size_final",
         **evolve_kwargs)

