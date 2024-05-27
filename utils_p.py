import time
import random
import os
import pickle
import numpy as np
from charles_p import Individual, Population
from selection_p import fps, tournament_selection
from mutation_p import polygon_mutation, pixel_mutation_random
from xo_p import blend_crossover, cut_crossover, pixel_crossover
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from operator import attrgetter
from skimage.transform import resize
from scipy import stats

target = plt.imread('scream.jpg')
img_shape = (25, 25)
target = resize(target, img_shape, preserve_range=True)

def phenotypic_entropy(self):

    """Calculates the phenotypic entropy of a population.

            Args:
                self: population of Individuals

            Returns:
                entropy: phenotypic_entropy
            """

    entropy = 0
    for individual in self:
        entropy += -individual.fitness*np.log(individual.representation)
    return entropy

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

def d(ind, target):

    delta_e_sum = 0
    for row in range(ind.representation.shape[0]):
        for pixel in range(ind.representation.shape[1]):
            delta_e_sum += delta_e(ind.representation[row][pixel], target.representation[row][pixel])
    return delta_e_sum

Individual.get_fitness = get_fitness
Population.get_entropy = phenotypic_entropy




#ind1 = Individual(shape=(25,25))
#ind2 = Individual(shape=(25,25))

#child1, child2 = cut_crossover(ind1.representation, ind2.representation, mirror_prob = 1)


#l = []
#print(ind1.fitness)
#for _ in range(1000):

    #ind2 = polygon_mutation(ind1,
                     #3,
                     #n_poly = [1,1],
                     #mutation_control = False,
                     #mutation_size = 3,
                     #init = False)
    #rep = pixel_mutation_random(ind1.representation, round(0.01*len(ind1.representation)*len(ind1.representation[0])), same_color = False)
    #ind2 = Individual(representation=rep)

    #a = d(ind1, ind2)
    #l.append(ind2.fitness)

#plt.hist(l)
#plt.show()

#plt.imshow(ind1.representation.astype(np.uint8))
#plt.title('ind1')
#plt.show()

search = 'selec_alg'
compiled_history_dir = os.path.join('grid', f'{search}', f'{search}_compiled_history.pkl')
best_individual_list_dir = os.path.join('grid', f'{search}', f'{search}_best_individual_list.pkl')
best_param_dir = os.path.join('grid', f'{search}', f'{search}_best_param.pkl')


with open(compiled_history_dir, 'rb') as file:
    compiled_history = pickle.load(file)

with open(best_individual_list_dir, 'rb') as file:
    best_individual_list = pickle.load(file)

with open(best_param_dir, 'rb') as file:
    best_param = pickle.load(file)

param_range = [0, 1]



#for param in range(len(best_individual_list)):
for param in range(len(param_range)):

    gen_avgs = []
    gen_stds = []
    for gen in range(len(best_individual_list[param])):
        gen_avgs.append(np.mean(best_individual_list[param][gen]))
        gen_stds.append(np.std(best_individual_list[param][gen]))

    plt.scatter(compiled_history[0][0][0],gen_avgs, label = str(param_range[param]), s = 5)

plt.minorticks_on()
plt.grid(which="minor", linestyle=":", linewidth=0.75)
plt.grid()
plt.title('Fitness History of best trials for each value')
plt.ylabel("Best Fitness")
plt.xlabel("Generation")
plt.legend()
plt.show()
plt.clf()









