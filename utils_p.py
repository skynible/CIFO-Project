import time
import random
import os
import pickle
import numpy as np
import re
import ast
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

Individual.get_fitness = get_fitness
Population.get_entropy = phenotypic_entropy




#ind1 = Individual(shape=(25,25), poly_range = [3, 5], vertices_range = [3, 5])
#ind2 = Individual(shape=(25,25))

#child1, child2 = cut_crossover(ind1.representation, ind2.representation, mirror_prob = 1)









searches = ['pop_size_final',
            'tour_size_final',
            'selec_alg_final',
            'mutation_alg_prob_final',
            'mut_prob_final',
            'xo_alg_prob_final',
            'mirror_prob_final',
            'xo_prob_final',
            'mutation_size_final']

best_avgs = []
best_sem = []
variance_avgs = []
variance_sem = []

# get history for each search
for search in searches:
    compiled_history_dir = os.path.join('grid', f'{search}', f'{search}_compiled_history.pkl')
    best_individuals_sample_dir = os.path.join('grid', f'{search}', f'{search}_best_individuals_sample.pkl')
    variance_sample_dir = os.path.join('grid', f'{search}', f'{search}_variance_sample.pkl')
    best_param_dir = os.path.join('grid', f'{search}', f'{search}_best_param.pkl')
    param_range_dir = os.path.join('grid', f'{search}', f'{search}.txt')

    with open(compiled_history_dir, 'rb') as file:
        compiled_history = pickle.load(file)

    with open(best_individuals_sample_dir, 'rb') as file:
        best_individuals_sample = pickle.load(file)

    with open(best_param_dir, 'rb') as file:
        best_param = pickle.load(file)

    # necessary patch due to different code versions
    if search == 'pop_size_final':
        variance_sample = compiled_history[best_param][:][3]
    else:
        with open(variance_sample_dir, 'rb') as file:
            variance_sample = pickle.load(file)

    if search == 'mutation_alg_prob_final':

        best_sample_avg = np.mean(best_individuals_sample[2][-1])
        best_sample_std = np.std(best_individuals_sample[2][-1])
        variance_sample_avg = np.mean(variance_sample[2][-1])
        variance_sample_std = np.std(variance_sample[2][-1])
        best_avgs.append(best_sample_avg)
        best_sem.append(best_sample_std / 10)
        variance_avgs.append(variance_sample_avg) # we divide by 10 because we are calculation the standard error of the mean and we did 100 trials
        variance_sem.append(variance_sample_std / 10)  # we divide by 10 because we are calculation the standard error of the mean and we did 100 trials

    else:

        best_sample_avg = np.mean(best_individuals_sample[best_param][-1])
        best_sample_std = np.std(best_individuals_sample[best_param][-1])
        variance_sample_avg = np.mean(variance_sample[best_param][-1])
        variance_sample_std = np.std(variance_sample[best_param][-1])
        best_avgs.append(best_sample_avg)
        best_sem.append(best_sample_std / 10) # we divide by 10 because we are calculation the standard error of the mean and we did 100 trials
        variance_avgs.append(variance_sample_avg)
        variance_sem.append(variance_sample_std / 10) # we divide by 10 because we are calculation the standard error of the mean and we did 100 trials

searches=[search[:-6] for search in searches]

plt.figure(figsize=(12, 8))
plt.errorbar(searches, best_avgs, marker='o', linestyle='-', color='b', yerr = best_sem)
plt.xticks(searches, rotation = 25, fontsize = 12)
plt.yticks(fontsize = 12)
plt.minorticks_on()
plt.xlabel('Parameter Tuned', fontsize = 12)
plt.ylabel('Last gen fitness sample average', fontsize = 12)
plt.title('Fitness History of Tuning Process', fontsize = 12)
plt.grid(which="minor", linestyle=":", linewidth=0.75)
plt.grid()
plt.show()
plt.clf()

plt.figure(figsize=(12, 8))
plt.errorbar(searches, variance_avgs, marker='o', linestyle='-', color='b', yerr = variance_sem)
plt.xticks(searches, rotation = 25, fontsize = 12)
plt.yticks(fontsize = 12)
plt.minorticks_on()
plt.xlabel('Average Variance last gen', fontsize = 12)
plt.ylabel('Last gen variance sample average', fontsize = 12)
plt.title('Variance History of Tuning Process', fontsize = 12)
plt.grid(which="minor", linestyle=":", linewidth=0.75)
plt.grid()
plt.show()


############################################################################################################


# searches = ['fs_sigma_final',
#             'fitness_sharing_final',
#             'xo_prob_final']
#
# avgs = []
# sem = []
# for search in searches:
#
#     best_individuals_sample_dir = os.path.join('grid', f'{search}', f'{search}_best_individuals_sample.pkl')
#     best_param_dir = os.path.join('grid', f'{search}', f'{search}_best_param.pkl')
#     param_range_dir = os.path.join('grid', f'{search}', f'{search}.txt')
#
#     with open(best_individuals_sample_dir, 'rb') as file:
#         best_individuals_sample = pickle.load(file)
#
#     with open(best_param_dir, 'rb') as file:
#         best_param = pickle.load(file)

    # with open(param_range_dir, 'r') as file:
    #     param_range_str = file.readlines()[2]
    #     match = re.search(r'\[.*?\]', param_range_str)
    #     list_str = match.group(0)
    #     param_range = ast.literal_eval(list_str)

#     if search == 'xo_prob_final':
#
#         best_sample_avg = np.mean(best_individuals_sample[best_param][199])
#         best_sample_std = np.std(best_individuals_sample[best_param][199])
#         avgs.append(best_sample_avg)
#         sem.append(best_sample_std / 10)
#
#     else:
#
#         best_sample_avg = np.mean(best_individuals_sample[best_param][-1])
#         best_sample_std = np.std(best_individuals_sample[best_param][-1])
#         avgs.append(best_sample_avg)
#         sem.append(best_sample_std / 10) # we divide by 10 because we are calculation the standard error of the mean and we did 100 trials
#
# plt.figure(figsize=(12, 8))
# plt.errorbar(searches, avgs, marker='o', linestyle='-', color='b', yerr = sem)
# plt.xticks(searches)
# plt.minorticks_on()
# plt.xlabel('Search Parameters')
# plt.ylabel('Values')
# plt.title('Values for Different Search Parameters')
# plt.grid(which="minor", linestyle=":", linewidth=0.75)
# plt.grid()
# plt.show()




 ############################################## SINGLE SEARCH #########################################################

# search = 'pop_size_final'
#
# compiled_history_dir = os.path.join('grid', f'{search}', f'{search}_compiled_history.pkl')
# best_individuals_sample_dir = os.path.join('grid', f'{search}', f'{search}_best_individuals_sample.pkl')
#entropy_sample_dir = os.path.join('grid', f'{search}', f'{search}_entropy_sample.pkl')
#variance_sample_dir = os.path.join('grid', f'{search}', f'{search}_variance_sample.pkl')
# best_param_dir = os.path.join('grid', f'{search}', f'{search}_best_param.pkl')
# param_range_dir = os.path.join('grid', f'{search}', f'{search}.txt')


# with open(compiled_history_dir, 'rb') as file:
#     compiled_history = pickle.load(file)
#
# with open(best_individuals_sample_dir, 'rb') as file:
#     best_individuals_sample = pickle.load(file)

# with open(entropy_sample_dir, 'rb') as file:
#     entropy_sample = pickle.load(file)
#
# with open(variance_sample_dir, 'rb') as file:
#     variance_sample = pickle.load(file)

# with open(best_param_dir, 'rb') as file:
#     best_param = pickle.load(file)
#
# with open(param_range_dir, 'r') as file:
#     param_range_str = file.readlines()[2]
#     match = re.search(r'\[.*?\]', param_range_str)
#     list_str = match.group(0)
#     param_range = ast.literal_eval(list_str)


#for param in range(len(best_individual_list)):
# for param in range(len(param_range)):
#
#     gen_avgs = []
#     gen_stds = []
#     for gen in range(len(best_individuals_sample[param])):
#         gen_avgs.append(np.mean(best_individuals_sample[param][gen]))
#         gen_stds.append(np.std(best_individuals_sample[param][gen]))

    #for param in range(len(compiled_history)):



    # plt.scatter(compiled_history[0][0][0],gen_avgs, label = str(param_range[param]), s = 5)
    #plt.scatter(range(0,200), gen_avgs, label=str(param_range[param]), s=5)

# plt.minorticks_on()
# plt.grid(which="minor", linestyle=":", linewidth=0.75)
# plt.grid()
# plt.title('Fitness History of best trials for each value')
# plt.ylabel("Best Fitness")
# plt.xlabel("Generation")
# plt.legend()
# plt.show()
# plt.clf()

# for param in range(len(param_range)):
#
#     gen_avgs = []
#     gen_stds = []
#     for gen in range(len(best_individuals_sample[param])):
#         gen_avgs.append(np.mean(entropy_sample[param][gen]))
#         gen_stds.append(np.std(entropy_sample[param][gen]))
#
#     #for param in range(len(compiled_history)):
#
#
#
#     plt.scatter(compiled_history[0][0][0],gen_avgs, label = str(param_range[param]), s = 5)
#
# plt.minorticks_on()
# plt.grid(which="minor", linestyle=":", linewidth=0.75)
# plt.grid()
# plt.title('Fitness History of best trials for each value')
# plt.ylabel("Best Fitness")
# plt.xlabel("Generation")
# plt.legend()
# plt.show()
# plt.clf()
#
# for param in range(len(param_range)):
#
#     gen_avgs = []
#     gen_stds = []
#     for gen in range(len(best_individuals_sample[param])):
#         gen_avgs.append(np.mean(variance_sample[param][gen]))
#         gen_stds.append(np.std(variance_sample[param][gen]))
#
#     #for param in range(len(compiled_history)):
#
#
#
#     plt.scatter(compiled_history[0][0][0],gen_avgs, label = str(param_range[param]), s = 5)
#
# plt.minorticks_on()
# plt.grid(which="minor", linestyle=":", linewidth=0.75)
# plt.grid()
# plt.title('Fitness History of best trials for each value')
# plt.ylabel("Best Fitness")
# plt.xlabel("Generation")
# plt.legend()
# plt.show()
# plt.clf()









