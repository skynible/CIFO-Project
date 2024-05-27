import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random

def polygon_mutation(Ind,
                     n_vertices,
                     n_poly = [1,2],
                     mutation_control = False,
                     mutation_size = 3,
                     init = False):

    """Mutation by adding a polygon to the image.

        Args:
            Ind: Individual or Individual.representation depending on init argument
            n_poly: number of polygons to be added to the image
            mutation_control: Bool that says if we strict the mutation size
            mutation_size: mutation size
            init: Bool that determines is argument Ind is Individual or Individual.representation

        Returns:
            Ind: either Individual or Individual.representation depending on init argument
        """

    if init:
        height, width, _ = Ind.shape
    else:
        height, width, _ = Ind.representation.shape

    if n_poly[-1] == 1:
        draw = 1
    else:
        draw = np.random.randint(n_poly[0], n_poly[1])
    for _ in range(draw):
        if mutation_control:
            # Select center of polygon and restrict the size with a square
            center = np.random.uniform(0, min(height, width), size=(2,))
            vertices = []

            # Consider the edges
            if center[0] - mutation_size<0:
                lower = 0
            else:
                lower = center[0] - mutation_size

            if center[0] + mutation_size>height:
                upper = center[0]
            else:
                upper = center[0] + mutation_size

            if center[1] - mutation_size<0:
                left = center[1]
            else:
                left = center[1] - mutation_size

            if center[1] + mutation_size>width:
                right = center[1]
            else:
                right = center[1] + mutation_size

            for _ in range(n_vertices):

                vertice = [np.random.uniform(left, right), np.random.uniform(lower, upper)]
                vertices.append(vertice)
        else:
            vertices = np.random.uniform(0, min(height, width), size=(n_vertices, 2))




        polygon_mask = np.zeros((height, width), dtype=bool)
        polygon = Polygon(vertices, closed=True)
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        coords = np.hstack((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)))
        mask = polygon.contains_points(coords)
        polygon_mask += mask.reshape(height, width)

        # Overlay the polygon on the original image
        if init:
            Ind = np.where(polygon_mask[..., None],
                                      [random.uniform(0, 255),
                                       random.uniform(0, 255),
                                       random.uniform(0, 255)],
                                       Ind)
        else:
            Ind.representation = np.where(polygon_mask[..., None],
                                      [random.uniform(0, 255),
                                       random.uniform(0, 255),
                                       random.uniform(0, 255)],
                                       Ind.representation)
    return Ind


def pixel_mutation_random(individual, n_pixels, same_color = True):

    """Mutation by randomly changing the color in random pixels.

            Args:
                Ind: Individual.representation
                n_pixels: number of pixels to be randomly selected and mutated
                same_color: boll that determins if the pixels change all to the same color or random ones

            Returns:
                Ind: Individual.representation
            """

    Ind = individual.copy()
    # randomly select pixels
    pixel_indices = []
    temp_row = np.random.randint(0, Ind.shape[0] - 1)
    temp_column = np.random.randint(0, Ind.shape[1] - 1)

    new_red = np.random.uniform(0, 255)
    new_green = np.random.uniform(0, 255)
    new_blue = np.random.uniform(0, 255)

    for _ in range(n_pixels):

        if not same_color:
            new_red = np.random.uniform(0, 255)
            new_green = np.random.uniform(0, 255)
            new_blue = np.random.uniform(0, 255)

        temp_row = np.random.randint(0, Ind.shape[0] - 1)
        temp_column = np.random.randint(0, Ind.shape[1] - 1)

        Ind[temp_row, temp_column, 0] = new_red
        Ind[temp_row, temp_column, 1] = new_green
        Ind[temp_row, temp_column, 2] = new_blue

    return Ind
