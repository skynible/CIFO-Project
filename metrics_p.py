import numpy as np

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