import os
import pickle
import math
import statistics
import time
from random import shuffle, choice, sample
import random
import numpy as np
from scipy import stats

from selection_p import fps, tournament_selection
from mutation_p import polygon_mutation, pixel_mutation_random
from xo_p import blend_crossover, cut_crossover, pixel_crossover

import matplotlib.pyplot as plt
from operator import attrgetter
from copy import copy
from selection_p import fps
from xo_p import blend_crossover
from skimage.transform import resize

################### BECAUSE OF FITNESS SHARING I ALSO DEFINE DELTA_E FUNCTION HERE #########################
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
#########################################################################################################
## we need to define the target in this document because of get_fitness function
target = plt.imread('scream.jpg')
img_shape = (25, 25)
target = resize(target, img_shape, preserve_range=True)
#############################################################################################################


class Individual:

    def __init__(self, representation=None, shape=None, poly_range = (1, 2), vertices_range = (3, 4)):

        #define the number of polygons randomly based on poly_range
        n_poly = np.random.randint(poly_range[0], poly_range[1])

        # fix background and add polygons
        if representation is None:
            self.representation=[]
            #self.shape = shape
            background = [random.uniform(0, 255), random.uniform(0, 255), random.uniform(0, 255)]

            # create fixed background color
            for _ in range(shape[0]):
                temp = [background for i in range(shape[1])]
                self.representation.append(temp)

            self.representation = np.array(self.representation)

            # add polygons
            for _ in range(n_poly):
                # define the number of vertices of each polygon
                n_vertices = np.random.randint(vertices_range[0], vertices_range[1])
                self.representation = polygon_mutation(self.representation, n_vertices, init = True)

        else:
            self.representation = representation

        self.fitness = self.get_fitness(target)

    def get_fitness(self, target):
        pass

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        return f" Fitness: {self.fitness}"


class Population:
    def __init__(self, size, optim, **kwargs):

        # population size
        self.size = size

        # defining the optimization problem as a minimization or maximization problem
        self.optim = optim

        self.kwargs = kwargs

        self.individuals = []

        # appending the population with individuals
        for _ in range(size):
            self.individuals.append(
                Individual(
                    shape = kwargs['shape'],
                    poly_range = kwargs['poly_range'],
                    vertices_range = kwargs['vertices_range']
                )
            )
        self.entropy = self.get_entropy()
        self.variance = self.get_variance()

    def evolve(self,
               gens, # int: number of generations
               selection, # list: list with selection functions [tournament_selection, fps] implemented
               selec_alg, # int: index of selection algorithm to be used
               tour_size, # size of tournament if used
               mutation, # list: list with mutation functions [polygon_mutation, pixel_mutation_random] implemented
               mut_prob, # float: mutation probability
               mutation_alg_prob, # float: probability of applying polygon_mutation instead of the other algorithm chosen
               pixel_mutation_same_color, # bool: determines if pixel_mutation_random uses the same color or random ones
               mut_vertices_range, # list: range of possible number of vertices of mutation polygon
               mut_poly_range, # list: range of possible number of polygons of mutation polygon
               mut_pixel_range, # list: range of possible number of pixels to be mutated
               crossover, # list: list with crossover algorithms [blend_crossover, cut_crossover, pixel_crossover] implemented
               xo_prob, # float: crossover probability
               xo_alg_prob, # list: list with crossover algorithms probability [blend_crossover_prob, cut_crossover_prob, pixel_crossover_prob] implemented
               mirror_prob, # float: [1,0] probability of mirroring the slices when applying cut_crossover
               elitism, # bool: determines if there is elitism or not
               fitness_sharing, # bool: determines if fitness sharing is used
               fs_sigma = 1, # float: niche radius for fitness sharing. Determines the amount of punish to clumps.
               early_stopping = None, # int: apply early stopping with int patience
               verbose = False):

        #initialize the history of the best_individual
        # history[0] = gens [0, 1, 2, 3, ....] history[1] = fitness [fit_ge1, fit_gen2, ...] history[2] = [entropy_gen1, ....] history[3] = [var_gen1, ...]
        history_best_individual_fitness = []
        history_pop_entropy = []
        history_pop_variance = []
        history_gens = []
        history = []

        # early stopping
        patience = 1
        best_fitness = 1e30


        for i in range(gens):

            start_time = time.time()
            new_pop = []

            elite = copy(min(self.individuals, key=lambda x: x.fitness))
            if elitism:
                new_pop.append(elite)


            #######################################################
            #  FITNESS SHARING ####################################
            #######################################################
            # fitness sharing implementation https://stackoverflow.com/questions/29734996/using-fitness-sharing-on-a-minimization-function
            fit = False
            if patience > 0:
                fit = True
            if fit:
                if fitness_sharing:
                    for ind1 in range(len(self)):
                        sharing_coefficient = 0
                        for ind2 in range(len(self)):
                            if ind1 != ind2:
                                delta_e_sum = 0
                                for row in range(self[ind1].representation.shape[0]):
                                    for pixel in range(self[ind1].representation.shape[1]):
                                        delta_e_sum += delta_e(self[ind1].representation[row][pixel],
                                                               self[ind2].representation[row][pixel])
                                # normalize
                                norm = 1 - (1 / (1 + delta_e_sum))
                                if norm < fs_sigma:
                                    sharing_function = 1 - (norm / fs_sigma)
                                    sharing_coefficient += sharing_function

                        if sharing_coefficient != 0:
                            self[ind1].fitness = self[ind1].fitness * sharing_coefficient

            ##########################################################################

            # check callback
            if early_stopping != None:
                if elite.fitness != best_fitness:
                    best_fitness = elite.fitness
                    patience = 1
                else:
                    patience += 1
                    if patience == early_stopping:
                        break

            ##################################### GEN LOOP ####################################

            while len(new_pop) < self.size:

                                            # SELECTION  AND CROSSOVER #

                # Choose selection algorithm
                if selec_alg == 0:
                    parent1, parent2 = selection[0](self, round(tour_size * self.size))

                if selec_alg == 1:
                    parent1, parent2 = selection[1](self)

                # Depending on the outcome of the draws we can have 1 or 2 offspring
                # so we need to take that into account with a double_offspring bool.
                # Since we also use more than one crossover algorithm with different
                # probabilities we create a list accumulated_prob that has the accumulated
                # probability intervals. We use crossover_done bool to not make more than
                # one crossover.

                double_offspring = False
                crossover_done = False
                accumulated_prob = []

                xo_draw = np.random.uniform()
                alg_draw = np.random.uniform()

                # getting the accumulated selection probability distribution of the population
                prob = 0
                for probability in xo_alg_prob:
                    prob += probability
                    accumulated_prob.append(prob)
                accumulated_prob[-1] = 1

                if xo_draw < xo_prob and alg_draw < accumulated_prob[0]: # so we use blend_crossover
                    offspring = crossover[0](parent1.representation, parent2.representation)
                    crossover_done = True

                elif xo_draw < xo_prob and alg_draw < accumulated_prob[1] and crossover_done == False: # so we use cut_crossover
                    offspring1, offspring2 = crossover[1](parent1.representation, parent2.representation, mirror_prob = mirror_prob)
                    double_offspring = True
                    crossover_done = True

                elif xo_draw < xo_prob and alg_draw < accumulated_prob[2] and crossover_done == False: # so we use cut_crossover
                    offspring = crossover[2](parent1.representation, parent2.representation)
                    crossover_done = True

                elif xo_draw > xo_prob:
                    offspring1 = parent1.representation
                    offspring2 = parent2.representation
                    double_offspring = True


                                            # MUTATION #

                if double_offspring:
                    # mutation with probability mut_prob
                    if np.random.uniform() < mut_prob:
                        # choosing mutation method with equal probability
                        draw = np.random.uniform()
                        if draw < mutation_alg_prob:
                            n_vertices = np.random.randint(mut_vertices_range[0], mut_vertices_range[1])
                            offspring1 = mutation[0](offspring1, n_vertices, n_poly = mut_poly_range, init = True)
                            n_vertices = np.random.randint(mut_vertices_range[0], mut_vertices_range[1])
                            offspring2 = mutation[0](offspring2, n_vertices, n_poly = mut_poly_range, init=True)
                        else:
                            n_pixels = np.random.randint(mut_pixel_range[0], mut_pixel_range[1])
                            offspring1 = mutation[1](offspring1, n_pixels, same_color = pixel_mutation_same_color)
                            n_pixels = np.random.randint(mut_pixel_range[0], mut_pixel_range[1])
                            offspring2 = mutation[1](offspring2, n_pixels, same_color = pixel_mutation_same_color)

                        new_pop.append(Individual(representation=offspring1))
                        if len(new_pop) < self.size:
                            new_pop.append(Individual(representation=offspring2))

                else:
                    # mutation with probability mut_prob
                    if np.random.uniform() < mut_prob:
                        # choosing mutation method with equal probability
                        draw = np.random.uniform()
                        if draw < mutation_alg_prob: #### ADD ARGUMENT FOR MUTATION_ALG_PROB IMPORTANT
                            n_vertices = np.random.randint(mut_vertices_range[0], mut_vertices_range[1])
                            offspring = mutation[0](offspring, n_vertices, n_poly = mut_poly_range, init = True)
                        else:
                            n_pixels = np.random.randint(mut_pixel_range[0], mut_pixel_range[1])
                            offspring = mutation[1](offspring, n_pixels, same_color = pixel_mutation_same_color)

                    new_pop.append(Individual(representation=offspring))


                                            # COMPILE #

            # assign new metrics
            self.individuals = new_pop
            self.entropy = self.get_entropy()
            self.variance = self.get_variance()
            if verbose:
                print('##########################################################################')
                print(f"Best individual of gen #{i + 1}: {min(self, key=lambda x: x.fitness)}")
                print(f"Population Entropy: {self.entropy}")
                print(f"Population Variance: {self.variance}")
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Execution time: {execution_time}")

            # append history of gen i+1
            history_best_individual_fitness.append(min(self, key=lambda x: x.fitness).fitness)
            history_pop_entropy.append(self.entropy)
            history_pop_variance.append(self.variance)
            history_gens.append(i + 1)


        # after last gen get the best individual and compile the history of the population
        best_individual = min(self, key=lambda x: x.fitness)

        history = [history_gens, history_best_individual_fitness, history_pop_entropy, history_pop_variance]
        #returns the history over the gen's and the best individual
        return history, best_individual


################################### GRID SEARCH EVOLUTION #####################################

    def grid(self,
             tuned_parameter, # string that will be used to create the save directory and determine which parameter to tune in kwargs
             param_range, # list that contains the range of values to grid search in the first index and the number of digits to round for [param_range, round]
             num_trials,
             alpha, # statistical significance threshold
             save_dir,
             **evolve_kwargs):

        mvp = Individual(shape=(img_shape))  # placeholder for the best individual after all parameters and trials
        mvp.fitness = 1e30
        compiled_history = [] # contains the history of the grid search compiled_history[param][trial][gens/fitness/entropy/variance history][gen]
                              # if we do compiled_history[param][trial][0][gen] we are selecting list of gens [0, 1, 2, 3, ...] (it is the same for all param, trial and gen) lazy
                              # if we do compiled_history[param][trial][1][gen] we are selecting the fitness history of the parameter param, in trial 'trial' at generation gen
                              # if we do compiled_history[param][trial][2][gen] we are selecting the entropy history ''      ''      ''      ''      ''      ''      ''      ''
                              # if we do compiled_history[param][trial][2][gen] we are selecting the variance history ''      ''      ''      ''      ''      ''      ''      ''

        best_individuals_list = [] # contains the best individuals by generation and parameter for every trial best_individuals_list[param][gen][trial]

        for param in range(len(param_range)):

            print('############################################')
            print(f'Starting parameter: {param_range[param]}')

            # defining the parameter
            if tuned_parameter != "pop_size":
                evolve_kwargs[tuned_parameter] = param_range[param]

            else:
                self.size = param_range[param]

            # it contains the trial history to be appended to compiled_history
            trial_history = []
            best_individuals = [[] for _ in range(evolve_kwargs['gens'])]

            for trial in range(num_trials):

                print(f'Doing trial: {param + 1}.{trial}')

                size = self.size
                optim = self.optim
                ind_kwargs = self.kwargs
                self.__init__(size, optim, **ind_kwargs)

                history, best_individual = self.evolve(gens = evolve_kwargs['gens'],
                                                      selection = evolve_kwargs['selection'],
                                                      selec_alg = evolve_kwargs['selec_alg'],
                                                      tour_size = evolve_kwargs['tour_size'],
                                                      mutation = evolve_kwargs['mutation'],
                                                      mut_prob = evolve_kwargs['mut_prob'],
                                                      mutation_alg_prob = evolve_kwargs['mutation_alg_prob'],
                                                      pixel_mutation_same_color = evolve_kwargs['pixel_mutation_same_color'],
                                                      mut_vertices_range = evolve_kwargs['mut_vertices_range'],
                                                      mut_poly_range = evolve_kwargs['mut_poly_range'],
                                                      mut_pixel_range = evolve_kwargs['mut_pixel_range'],
                                                      crossover = evolve_kwargs['crossover'],
                                                      xo_prob = evolve_kwargs['xo_prob'],
                                                      xo_alg_prob = evolve_kwargs['xo_alg_prob'],
                                                      mirror_prob = evolve_kwargs['mirror_prob'],
                                                      elitism = evolve_kwargs['elitism'],
                                                      fitness_sharing = evolve_kwargs['fitness_sharing'],
                                                      fs_sigma = evolve_kwargs['fs_sigma'],
                                                      early_stopping = evolve_kwargs['early_stopping'],
                                                      verbose = evolve_kwargs['verbose'])

                for gen in range(len(best_individuals)):
                    best_individuals[gen].append(history[1][gen])

                # get best overall individual
                if best_individual.fitness < mvp.fitness:
                    mvp.fitness = best_individual.fitness
                    mvp.representation = best_individual.representation

                trial_history.append(history)

            best_individuals_list.append(best_individuals)
            compiled_history.append(trial_history)

        ################### COMPARE THE BEST INDIVIDUAL AVERAGE OF EACH PARAMETER VALUE ####################

        save_dir_c = os.path.join('grid', f'{save_dir}')
        os.makedirs(save_dir_c, exist_ok=True)

        best_last_gen_param_avg = 1e30
        best_last_gen_param = 0
        KS_test_results = []

        for param in range(len(best_individuals_list)):

            # getting the avgs and stds of best individuals in each generation
            gen_avgs = []
            gen_stds = []
            for gen in range(len(best_individuals_list[param])):
                gen_avgs.append(np.mean(best_individuals_list[param][gen]))
                gen_stds.append(np.std(best_individuals_list[param][gen]))

            # get the param that had the best avg in the last generation
            if gen_avgs[-1] < best_last_gen_param_avg:
                best_last_gen_param_avg = gen_avgs[-1]
                best_last_gen_param = param


            ############################## TESTING FOR SAMPLE NORMALITY #################################

            # Performing a Kolmogorov–Smirnov test on last generation of each param to see
            # if the data is not normally distributed.

            # First we have to fit a normal function to the data to feed to stats.kstest
            loc, scale = stats.norm.fit(best_individuals_list[param][-1])
            normal = stats.norm(loc = loc, scale = scale)

            # Performing the Kolmogorov–Smirnov test
            KS_test = stats.kstest(best_individuals_list[param][-1], normal.cdf)
            KS_test_pvalue = KS_test[1]

            # if the p-value is bigger than alpha we do not have statistical evidence that
            # the sample is not normally distributed so we are going to assume it is
            if KS_test_pvalue > alpha:
                KS_test_results.append(True)

            # if the p-value is smaller than alpha we have statistical evidence to say
            # the sample is not normally distributed (at an alpha significance level)
            else:
                KS_test_results.append(False)


        ########################### SIGNIFICANCE TESTS AND TXT COMPILATION ###########################

        lines = [] # where we store the lines to be written to the txt file that saves information of the grid
        lines.append(f"Population size: {self.size} \n")
        lines.append(f"Tuned parameter: {tuned_parameter} \n")
        lines.append(f"range of values tested: {param_range} \n")
        lines.append(f"Number of trials: {num_trials} \n\n")

        lines.append(f"Evolution function kwargs: \n")
        lines.append(f"\t gens: {evolve_kwargs['gens']} \n")
        lines.append(f"\t selection: {evolve_kwargs['selection']} \n")
        lines.append(f"\t selec_alg: {evolve_kwargs['selec_alg']} \n")
        lines.append(f"\t tour_size: {evolve_kwargs['tour_size']} \n")
        lines.append(f"\t mutation: {evolve_kwargs['mutation']} \n")
        lines.append(f"\t mut_prob: {evolve_kwargs['mut_prob']} \n")
        lines.append(f"\t mutation_alg_prob: {evolve_kwargs['mutation_alg_prob']} \n")
        lines.append(f"\t pixel_mutation_same_color: {evolve_kwargs['pixel_mutation_same_color']} \n")
        lines.append(f"\t mut_vertices_range: {evolve_kwargs['mut_vertices_range']} \n")
        lines.append(f"\t mut_poly_range: {evolve_kwargs['mut_poly_range']} \n")
        lines.append(f"\t mut_pixel_range: {evolve_kwargs['mut_pixel_range']} \n")
        lines.append(f"\t crossover: {evolve_kwargs['crossover']} \n")
        lines.append(f"\t xo_prob: {evolve_kwargs['xo_prob']} \n")
        lines.append(f"\t xo_alg_prob: {evolve_kwargs['xo_alg_prob']} \n")
        lines.append(f"\t mirror_prob: {evolve_kwargs['mirror_prob']} \n")
        lines.append(f"\t elitism: {evolve_kwargs['elitism']} \n")
        lines.append(f"\t fitness_sharing: {evolve_kwargs['fitness_sharing']} \n")
        lines.append(f"\t fs_sigma: {evolve_kwargs['fs_sigma']} \n")
        lines.append(f"\t early_stopping: {evolve_kwargs['early_stopping']} \n\n")

        lines.append("Individual kwargs: \n")
        lines.append(f"\t shape: {self.kwargs['shape']} \n")
        lines.append(f"\t poly_range: {self.kwargs['poly_range']} \n")
        lines.append(f"\t vertices_range: {self.kwargs['vertices_range']} \n\n")

        lines.append(f"Best {tuned_parameter} found: {param_range[best_last_gen_param]} \n\n")

        lines.append("Significance testing: \n\n")

        # checking if we can assume that the sample of the param that had the best avg in the last generation
        # is normally distributed. If so we apply a t-student test of two samples to see if
        # the avg of the best performing sample is equal or different from the rest of the samples.
        # We also do a t-student test to see if the avg of the best performing sample is in fact
        # smaller than the rest of the samples. To perform both of these tests we need to see if we
        # have statistical evidence to say that the samples we are going to compare don't have equal variance.

        if KS_test_results[best_last_gen_param]:
            for param in range(len(KS_test_results)):
                if param != best_last_gen_param:

                    lines.append(f"Tests comparing {param_range[best_last_gen_param]} sample with {param_range[param]} sample: \n")

                    if KS_test_results[param]:

                        var_test_result = stats.levene(best_individuals_list[best_last_gen_param][-1],
                                                        best_individuals_list[param][-1])
                        lines.append(f"\t Levene test p-value: {var_test_result.pvalue} \n")

                        # we have statistical evidence to say that the two samples come from populations
                        # with different variances. So we apply the two sample t-student for different variances
                        if var_test_result.pvalue < alpha:
                            eq_avg_test_result = stats.ttest_ind(best_individuals_list[best_last_gen_param][-1],
                                                                 best_individuals_list[param][-1],
                                                                 equal_var = False,
                                                                 alternative = 'two-sided')

                            ineq_avg_test_result = stats.ttest_ind(best_individuals_list[best_last_gen_param][-1],
                                                                 best_individuals_list[param][-1],
                                                                 equal_var=False,
                                                                 alternative='less')
                            lines.append(f"\t Average equality t-student test p-value: {eq_avg_test_result[1]} \n")
                            lines.append(f"\t Average inequality t-student test p-value: {ineq_avg_test_result[1]} \n")

                        # we do not have statistical evidence to say that the two samples come from
                        # populations with different variance so we are going to assume that they
                        # came from populations with equal variance
                        else:
                            eq_avg_test_result = stats.ttest_ind(best_individuals_list[best_last_gen_param][-1],
                                                                 best_individuals_list[param][-1],
                                                                 equal_var = True,
                                                                 alternative = 'two-sided')
                            ineq_avg_test_result = stats.ttest_ind(best_individuals_list[best_last_gen_param][-1],
                                                                 best_individuals_list[param][-1],
                                                                 equal_var=True,
                                                                 alternative='less')
                            lines.append(f"\t Average equality t-student test p-value: {eq_avg_test_result[1]} \n")
                            lines.append(f"\t Average inequality t-student test p-value: {ineq_avg_test_result[1]} \n")

                    # if the sample is not normally distributed we perform a Wilcoxon Rank-Sum Test
                    # to see if the population corresponding to the best sample has in fact a distribution
                    # that has a smaller average than the distribution of the sample in question.
                    else:
                        lines.append(f'\t parameter {param_range[param]} not normal distributed. \n')
                        rank_test_result = stats.ranksums(best_individuals_list[best_last_gen_param][-1],
                                                          best_individuals_list[param][-1],
                                                          alternative = 'less')
                        lines.append(f"\t Wilcoxon Rank-Sum Test p-value: {rank_test_result[1]} \n")

        # if the best sample is not normally distributed we perform a Wilcoxon Rank-Sum Test
        # to see if the population corresponding to the best sample has in fact a distribution
        # that has a smaller average than the distributions of the other samples.
        else:
            lines.append(f'Best parameter {param_range[best_last_gen_param]} not normal distributed. \n')

            for param in range(len(KS_test_results)):
                if param != best_last_gen_param:

                    lines.append(f"Tests comparing {param_range[best_last_gen_param]} sample with {param_range[param]} sample: \n")
                    rank_test_result = stats.ranksums(best_individuals_list[best_last_gen_param][-1],
                                                      best_individuals_list[param][-1],
                                                      alternative='less')
                    lines.append(f"\t Wilcoxon Rank-Sum Test p-value: {rank_test_result[1]} \n")

        # Write all lines to the txt file
        txt_dir = os.path.join('grid', f'{save_dir}', f'{save_dir}.txt')
        with open(txt_dir, 'w') as file:
            file.writelines(lines)

        # make pickle objects with information to make plots
        compiled_history_pickle_dir = os.path.join('grid', f'{save_dir}', f'{save_dir}_compiled_history.pkl')
        with open(compiled_history_pickle_dir, 'wb') as file:
            pickle.dump(compiled_history, file)

        best_individual_list_pickle_dir = os.path.join('grid', f'{save_dir}', f'{save_dir}_best_individual_list.pkl')
        with open(best_individual_list_pickle_dir, 'wb') as file:
            pickle.dump(best_individuals_list, file)

        best_param_pickle_dir = os.path.join('grid', f'{save_dir}', f'{save_dir}_best_param.pkl')
        with open(best_param_pickle_dir, 'wb') as file:
            pickle.dump(best_last_gen_param, file)

        ########################### BEST INDIVIDUAL ###########################

        plt.imshow(mvp.representation.astype(np.uint8))
        plt.title(f'Best_result_{tuned_parameter}')
        save_path = os.path.join(save_dir_c, f'best_individual_{tuned_parameter}.png')
        plt.savefig(save_path)
        plt.show()
        plt.clf()


    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]
    def get_entropy(self):
        pass
    def get_variance(self):
        pass
