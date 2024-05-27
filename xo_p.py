from random import randint
import numpy as np


def blend_crossover(parent1, parent2):

    """ blending parents with opacity x for parent1 and x-1 for parent2.

        Args:
            parent1: Individual.representation
            parent2: Individual.representation

        Returns:
            child: Individual.representation of the child
        """

    #sample opacity for extra stochasticity
    x = np.random.uniform(0,1)
    child = (parent1 * x) + (parent2 * (1 - x))
    return child

def cut_crossover(parent1, parent2, mirror_prob = 0):

    """ blending parents by cutting image in random direction and position.

        Args:
            parent1: Individual.representation
            parent2: Individual.representation
            mirror_prob: float [0,1] probability of mirroring

        Returns:
            child1: Individual.representation of the child1
            child2: Individual.representation of the child2
        """

    cut_index = np.random.randint(1, parent1.shape[0]-1)
    #print('cut_index:', cut_index)
    #print('parent1:',parent1.shape)
    #print('parent2:', parent2.shape)

    #randomly select if the cut is vertical or horizontal
    draw = np.random.uniform()
    if draw < 0.5:



        parent1_top = parent1[:cut_index+1, :, :]
        parent1_bottom = parent1[cut_index +1:, :, :]
        parent2_top = parent2[:cut_index+1, :, :]
        parent2_bottom = parent2[cut_index+1:, :, :]

        if np.random.uniform() < mirror_prob:
            parent1_top = parent1_top[::-1]

        if np.random.uniform() < mirror_prob:
            parent1_bottom = parent1_bottom[::-1]

        if np.random.uniform() < mirror_prob:
            parent2_top = parent2_top[::-1]

        if np.random.uniform() < mirror_prob:
            parent2_bottom = parent2_bottom[::-1]

        child1 = np.concatenate((parent1_top, parent2_bottom), axis = 0)
        child2 = np.concatenate((parent2_top, parent1_bottom), axis = 0)

    if draw > 0.5:



        parent1_left = parent1[:, :cut_index + 1, :]
        parent1_right = parent1[:, cut_index + 1:, :]
        parent2_left = parent2[:, :cut_index + 1, :]
        parent2_right = parent2[:, cut_index +1:, :]

        if np.random.uniform() < mirror_prob:
            parent1_left = [row[::-1] for row in parent1_left]

        if np.random.uniform() < mirror_prob:
            parent1_right = [row[::-1] for row in parent1_right]

        if np.random.uniform() < mirror_prob:
            parent2_left = [row[::-1] for row in parent2_left]

        if np.random.uniform() < mirror_prob:
            parent2_right = [row[::-1] for row in parent2_right]

        child1 = np.concatenate((parent1_left, parent2_right), axis = 1)
        child2 = np.concatenate((parent2_left, parent1_right), axis = 1)

    #print('child1:',child1.shape)
    #print('child2:', child2.shape)

    return child1, child2

def pixel_crossover(parent1, parent2):
    """ blending parents by cutting image in random direction and position.

       Args:
           parent1: Individual.representation
           parent2: Individual.representation

       Returns:
           child: Individual.representation of child
       """

    child = parent1.copy()
    for row in range(len(parent1[0])):
        for col in range(len(parent1[1])):
            draw = np.random.uniform()
            if draw < 0.5:
                child[row][col] = parent2[row][col]
    return child


def orgy_xo(ind_list):
    R = np.random.uniform(0, 1, len(ind_list))
    total_r = np.sum(R)
    R = R / total_r
    total_r2 = np.sum(R)
    child = np.zeros(ind_list[0].representation.shape)
    for i in range(len(ind_list)):
        child += ind_list[i]*R[i]
    return child



