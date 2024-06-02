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

##########################################################
target = plt.imread('scream.jpg')
img_shape = (25, 25)
target = resize(target, img_shape, preserve_range=True)
##########################################################

####################################### COMPILATION #############################################################

# monkey patch functions
Individual.get_fitness = get_fitness
Population.get_variance = phenotypic_variance


kwargs = {
    'shape': img_shape, # Individual shape
    'poly_range': [2, 3], # possible number of polygons when starting the individual
    'vertices_range': [3, 5] # possible number of vertices  of polygons
}

# we did not check if the edges of the same polygon do not intercept

pop_size = 10

pop = Population(size = pop_size, optim = 'min', **kwargs)
history, best_individual = pop.evolve(gens = 70000,
                                      selection = [tournament_selection, fps],
                                      selec_alg = 1,
                                      tour_size = 0.1,
                                      mutation = [polygon_mutation, pixel_mutation_random],
                                      mut_prob = 0.04,
                                      mutation_alg_prob = 1,
                                      pixel_mutation_same_color = True,
                                      mut_vertices_range = [3,5],
                                      mut_poly_range = [1,1],
                                      mut_pixel_range = [img_shape[0]*img_shape[1]*0.005, img_shape[0]*img_shape[1]*0.01],
                                      crossover = [blend_crossover, cut_crossover, pixel_crossover],
                                      xo_prob = 0.8,
                                      xo_alg_prob = [0.7, 0.25, 0.05],
                                      mirror_prob = 0.05,
                                      elitism = True,
                                      fitness_sharing = False,
                                      fs_sigma = 1,
                                      mutation_size = None,
                                      early_stopping = 2000,
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

