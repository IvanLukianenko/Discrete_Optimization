#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
from collections import namedtuple
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
Point = namedtuple("Point", ['x', 'y'])

import opt2

def length(point1, point2):
    """
        Вычисление евклидового расстояния.
    """
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def create_table(nodeCount, points):
    """
        Создание матрицы расстояний.
    """
    a = [[0] * (nodeCount) for i in range(nodeCount)]
    a = np.array(a)
    for i in range(nodeCount):
        for j in range(nodeCount):
            a[i][j] = length(points[i],points[j])
    return a

def find_nearest_point(index, d_m, set_of_points):
    """
        Поиск ближайшей точки.
    """
    min_path = 200000
    index_of_min = -1
    for j in set_of_points:
        if d_m[index][j] < min_path and j!=index:
            min_path = d_m[index][j]
            index_of_min = j
    if(index_of_min == -1):
        return index, []
    else:
        set_of_points.remove(index_of_min)
        return index_of_min, set_of_points

def greedy(s, d_m, points):
    """
        Жадный алгоритм для tsp.
    """
    starting_point = s
    set_of_points = list(range(len(d_m[0])))
    set_of_points.remove(s)
    actual_point, set_of_points = find_nearest_point(starting_point, d_m, set_of_points)
    solution = []
    solution.append(starting_point)
    solution.append(actual_point)

    i = 2
    while i < len(d_m[0]):
        actual_point, set_of_points = find_nearest_point(actual_point, d_m, set_of_points)
        solution.append(actual_point)
        i += 1
    return solution

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')
    points = []
    nodeCount = int(lines[0])
    tabu_list = [(0,0), (0,0), (0,0)]
    
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    solution = [i for i in range(nodeCount)]

    if nodeCount < 20000:
        s = random.randint(0, nodeCount-1)
        distance_matrix = create_table(nodeCount, points)
        solution = greedy(s, distance_matrix, points)
    else:
        solution = [i for i in range(nodeCount)]
        random.shuffle(solution)

    solution = opt2.two_opt(solution, points)

    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

