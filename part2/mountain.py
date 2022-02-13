#!/usr/local/bin/python3
#
# Authors: admysore-hdeshpa-machilla
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, April 2021
#
import math

from PIL import Image
from numpy import *
# from numpy.core import amax
from numpy.ma import amax, argmax, argmin
from scipy.ndimage import filters
import sys
import imageio
from numpy import array, zeros, sqrt, where, log

# Class which has stored the transition and the probability table and backtracking table
class mountain:
    # What should your transition probabilities look like? It’s up to you, but you probably want to
    # use a distribution that encourages “smoothness,” i.e. that P(si+1|si) is high if si+1 = si and is low otherwise
    # We assumed transition probability values
    # the pixel in same row has high chance of being in the ridgeLine so I have kept the transition probability as 0.9
    # if pixel is in a immediate row above or below
    # we keep on decreasing the transition probabilities as we go away from the pixel which we are considering
    # for all the other rows we will not calculate their probabilities as the chance of them being in the ridge line are low


        transition_probabilities = [0.9, 0.75, 0.2, 0.01, 0.005, 0.0004]
        # Array which will stores the probabilities ( Default Initialization)
        pixel_probability_table = zeros((10000,10000))
        # Array to store all the y indexes( row index) of every column where we found the maximum probability ( Default Initialization)
        backtracking_table = zeros((10000,10000))

# calculate "Edge strength map" of an image
#
def edge_strength(input_image):

    grayscale = array(input_image.convert('L'))

    filtered_y = zeros(grayscale.shape)

    filters.sobel(grayscale,0,filtered_y)

    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image


# Getting an array which will store the probabilities for every pixel.
# In this function we will only get the probability of first column (so all its rows) in the array
def get_initial_pixel_probability(edge_strength, total_row_len, total_col_len):

    # Array which will stores the probabilities
    # Re initializing the array to the size of the image we have
    mountain_obj.pixel_prob = zeros(edge_strength.shape)
    initial_pixel_prob = mountain_obj.pixel_probability_table

    # sums up the edge strength values of first column of the image
    column_total = sum(edge_strength[0:total_row_len, 0])


    # calculating the initial probabilities. i.e Only for first column of the image
    for row in range(total_row_len):
        initial_pixel_prob[row][0] = (edge_strength[row][0] / column_total)

    return  initial_pixel_prob

# Calculating probability of each pixel where we have set a value for checking only a few rows up and
# below the pixel to check its worth of being a Earth Sky horizon
# Probability = Initial probability * transition probability * emission probability
def get_probability(edge_strength, probability_table, previous_max_pixel_table , total_row_len, total_col_len, row_start, row_end, row_step,
                          col_start,  col_end, col_step, transition_start):

    for column in range(col_start,  col_end, col_step):
        for rows in range(row_start, row_end, row_step):
            maximum_intensity_pixel_prob = 0
            # Loop keps track of rows which I need to scan. So we scan only top and botton 5 rows
            for immediate_row in range(-transition_start, transition_start + 1, 1):
                if ((rows + immediate_row < total_row_len) & (rows + immediate_row >= 0)):
                    # We have kept the emission probability as the light intensity value from the edge strength matrix
                    emission_prob = edge_strength[rows][column]
                    pixel_probability = (probability_table[rows + immediate_row][column - col_step]) * (mountain_obj.transition_probabilities[abs(immediate_row)])
                    if (maximum_intensity_pixel_prob < pixel_probability):
                        maximum_intensity_pixel_prob = pixel_probability
                        # array which will be useful in backtracking for the solution,
                        # this keeps track of the row indexes at which we found the maximum intensity in a column
                        previous_max_pixel_table[rows][column] = rows + immediate_row

                    probability_table[rows][column] = (emission_prob/1000) * (maximum_intensity_pixel_prob)

    return (probability_table, previous_max_pixel_table )


# Function is useful in backtracking to the previous columns maximum value( i.e the row at which the maximum intensity was captured)
# which is stored in array during the probability calculations
def backtracking(ridge_line, maximum_intensity_pixel_prob, previous_max_pixel_table, total_col_len):
    # We scan from right to left of the image
    for column in range(total_col_len - 1, -1, -1):
        ridge_line[column] = int(maximum_intensity_pixel_prob)
        # We have stored the row index in below table to keep track of the maximum intensity row
        maximum_intensity_pixel_prob = previous_max_pixel_table[int(maximum_intensity_pixel_prob)][column]

    return (ridge_line, previous_max_pixel_table)


# We follow the viterbi algorith where we calculate all the probabilities of pixel's and store their values in a table.
# We use this dynamic programming methodology to reduce recalculations of probabilities.
# Find the maximum probability for the last column of image( i.e scan all the rows of the last column)
# and find the index( i.e the row index) at which we found the maximum value.
# Using this index, we backtrack to the first column of image backtrack to the solutions
def viterbi(edge_strength, initial_pixel_prob, previous_max_pixel_table, ridge, total_row_len, total_col_len, row_start, row_end, row_step,
                          col_start,  col_end, col_step, transition_start ):


    (probability_table, previous_max_pixel_table) = get_probability(edge_strength, initial_pixel_prob,
                                                                               previous_max_pixel_table, total_row_len,
                                                                               total_col_len, row_start, row_end, row_step,
                                                                            col_start,  col_end, col_step, transition_start)

    # Index (row) of Maximum intensity pixel of the last column. So that we can backtrack from right to left of the image
    maximum_intensity_pixel_prob = argmax(probability_table[0:total_row_len, total_col_len - 1])

    (ridge, previous_max_pixel_table) = backtracking(second_ridge, maximum_intensity_pixel_prob,
                                                            previous_max_pixel_table, total_col_len)

    return (ridge, probability_table, previous_max_pixel_table)


# We find the row index at which we get maximum intensity of light for all the columns of the image
def bayes_net(edge_strength, total_row_len, total_col_len):
    ridge_line = []
    # (total_row_len, total_col_len) = edge_strength.shape
    for i in range(0, total_col_len):
        # Getting all the values of a column in a list
        single_row = edge_strength[0:total_row_len, i].flatten()
        single_row = list(single_row)
        # Getting the maximum value ( intensity value) of a single column and getting the index (row index)
        # at which it will have highest intensity in the image
        highest_intensity_col = single_row.index(max(single_row))
        # Appending this in a list to keep track of the row co-ordinates( y)
        ridge_line.append(highest_intensity_col)

    # imageio.imwrite("output_first.jpg", draw_edge(input_image, ridge_line, (255, 0, 0), 5))
    return ridge_line


# main program
#
gt_row = -1
gt_col = -1
if len(sys.argv) == 2:
    input_filename = sys.argv[1]
elif len(sys.argv) == 4:
    (input_filename, gt_row, gt_col) = sys.argv[1:]

else:
    raise Exception("Program requires either 1 or 3 parameters")

# load in image
input_image = Image.open(input_filename)

# Changing the type from str class to int
gt_row, gt_col = int(gt_row), int(gt_col)


# compute edge strength mask
edge_strength = edge_strength(input_image)



imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))

#
#
# First Solution : Bayes Net

(total_row_len, total_col_len) = edge_strength.shape
ridge_line = bayes_net(edge_strength, total_row_len, total_col_len)


imageio.imwrite("output_first.jpg", draw_edge(input_image, ridge_line, (255, 0, 0), 5))

#Second Solution : Viterbi Approach

# Class which has stored the transition and the probability table
mountain_obj = mountain()
initial_pixel_prob = get_initial_pixel_probability(edge_strength, total_row_len, total_col_len)
# Initializing the table which will keep track of all the row indexes where we found the maximum probability of being a ridge
# Used for backtracking
mountain_obj.backtracking_table = zeros(edge_strength.shape)
previous_max_pixel_table = mountain_obj.backtracking_table

second_ridge = zeros(total_col_len)

# We will traverse for only these rows while calculating the probabailities.
# We dont need to travell all the rows of a column as the ridge will need to be a continous line.
# The probability of finding a ridge in the 0th row when in the previous column's pixel we found it at 50th row is very low
transition_rows = len(mountain_obj.transition_probabilities)-1

(second_ridge_final, probability_table, previous_max_pixel_table) = viterbi(edge_strength, initial_pixel_prob,
                                                                            previous_max_pixel_table, second_ridge,
                                                                            total_row_len, total_col_len,
                                                                            0, total_row_len, 1,
                                                                            1, total_col_len, 1, transition_rows )
# This will draw the edge on the original image
input_image = Image.open(input_filename)

imageio.imwrite("output_second.jpg", draw_edge(input_image, second_ridge, (0, 0, 255), 5))

# Third Solution: Taking human input for rows and Columns With viterbi
third_ridge = [total_row_len /2] * total_col_len


# using human input values to reset the probabilities
# probability_table_back = probability_table.copy()
# previous_max_pixel_table_back = previous_max_pixel_table.copy()
probability_table[0:total_row_len,gt_col] = 0
probability_table[gt_row][gt_col] = 1

# checking and calculating the probabilities in forward motion
(third_ridge, probability_table_forw, previous_max_pixel_table_forw) = viterbi(edge_strength, probability_table,
                                                                                     previous_max_pixel_table,
                                                                                     third_ridge, total_row_len,
                                                                                     total_col_len,
                                                                                     0, total_row_len, 1,
                                                                                     gt_col + 1, total_col_len, 1,
                                                                                     transition_rows)
# checking and calculating the probabilities in backward motion
(third_ridge, probability_table, previous_max_pixel_table) = viterbi(edge_strength, probability_table,
                                                                                    previous_max_pixel_table, third_ridge,
                                                                                    total_row_len, total_col_len,
                                                                                    (total_row_len - 1), -1, -1,
                                                                                    (gt_col - 1), 0, -1,
                                                                                    transition_rows)



# This will draw the edge on the original image
input_image = Image.open(input_filename)
imageio.imwrite("output_third.jpg", draw_edge(input_image, third_ridge, (0, 255, 0), 5))
