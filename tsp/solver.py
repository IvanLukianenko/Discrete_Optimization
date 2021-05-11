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
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def create_table(nodeCount, points):
    a = [[0] * (nodeCount) for i in range(nodeCount)]
    a = np.array(a)
    for i in range(nodeCount):
        for j in range(nodeCount):
            a[i][j] = length(points[i],points[j])
    return a

def find_nearest_point(index, d_m, set_of_points):
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

def reverse_segment_if_better(tour, i, j, k, t, points):
    """If reversing tour[i:j] would make the tour shorter, then do it."""
    #Дан путь - [...A-B...C-D...E-F...]
    A, B, C, D, E, F = points[tour[i-1]], points[tour[i]], points[tour[j-1]], points[tour[j]], points[tour[k-1]], points[tour[k % len(tour)]]
    d0 = length(A, B) + length(C, D) + length(E, F)
    d1 = length(A, C) + length(B, D) + length(E, F)
    d2 = length(A, B) + length(C, E) + length(D, F)
    d3 = length(A, D) + length(E, B) + length(C, F)
    d4 = length(F, B) + length(C, D) + length(E, A)

    deltas = [d1 - d0, d2 - d0, d3 - d0, d4 - d0]

    min_ = min(deltas)
    #random.random() - вернет вероятность от 0 до 1 
    if min_ >= 0:   
        #if random.random() > np.exp(-min_/max(t, 0.0000001)):
            #print("Не Отжигаю")
        #    return 0
        #else:
            #print("Отжигаю")
        #    pass
        return 0
    if deltas[2] == min_:
        tmp = tour[j:k] + tour[i:j]
        tour[i:k] = tmp
        return deltas[2]
    elif deltas[0] == min_:
        tour[i:j] = reversed(tour[i:j])
        return deltas[0]
    elif deltas[1] == min_:
        tour[j:k] = reversed(tour[j:k])
        return deltas[1]
    elif deltas[3] == min_:
        tour[i:k] = reversed(tour[i:k])
        return deltas[3]

def three_opt(tour, points):
    """Iterative improvement based on 3 exchange."""
    big_delta = 0
    t = 100
    k = 0
    while True:
        delta = 0
        for (a, b, c) in all_3segments(len(tour)):
            delta += reverse_segment_if_better(tour, a, b, c, t, points)
            t = 0.9*t
        if delta >= 0:
            break 
        k += 1
        if len(tour) >= 500:
            if k == 1: 
                break
    return tour
def reverse_segment_if_2better(tour, i, j, t, points, tabu_list):
    """If reversing tour[i:j] would make the tour shorter, then do it."""
    #Дан путь - [...A-B...C-D...]
    A, B, C, D = points[tour[i-1]], points[tour[i]], points[tour[j-1]], points[tour[j % len(tour)]]
    d0 = length(A, B) + length(C, D)
    d1 = length(A, C) + length(B, D)
    d2 = length(A, D) + length(B, C)

    delta = d1 - d0

    
    #random.random() - вернет вероятность от 0 до 1 
    if delta >= 0:   
        #if random.random() >= np.exp(-delta/max(t, 0.0000001)):
        #    print("Не Отжигаю")
            #return 0
        #    print("Отжигаю")
        return 0
    #print(delta)
    if (i, j) in tabu_list or (j, i) in tabu_list:
        return 0
    tour[i:j] = reversed(tour[i:j])
    tabu_list.append((i,j))
    if len(tabu_list) >= 3:
        del tabu_list[0]
    return delta
   

def all_3segments(n: int):
    """Generate all segments 3-combinations"""
    return ((i, j, k)
        for i in range(n)
        for j in range(i + 2, n)
        for k in range(j + 2, n + (i > 0)))

def two_opt(tour, points, tabu_list):
    """Iterative improvement based on 2 exchange."""
    big_delta = 0
    t = 100000
    t0 = 10
    k = 0
    while True:
        delta = 0
        for (a, b) in all_2segments(len(tour)):
            delta += reverse_segment_if_2better(tour, a, b, t, points, tabu_list)
            t = 0.7 * t 
        if delta >= 0:
            big_delta += 1
            if big_delta >= 10:
                break 
        k += 1
        if len(tour) >= 500:
            if k == 1: 
                break
    return tour

def all_2segments(n: int):
    """Generate all segments 2-combinations"""
    return ((i, j)
        for i in range(n)
        for j in range(i + 2, n + (i>0)))

def tour_length(solution, points, nodeCount):
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])
    return obj

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

    # creating distance matrix
    #distance_matrix = create_table(nodeCount, points)

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    #solution1 = None
    #solutions = []
    #solution = [i for i in range(nodeCount)]
    #solution1 = solution
    #for i in range(5):
    #    solution = three_opt(solution1, points, tabu_list)
    #    solutions.append([solution, tour_length(solution, points, nodeCount)])
    #    random.shuffle(solution1)
    best = None
    value = 100000000
    #if nodeCount <= 1900:
    #    distance_matrix = create_table(nodeCount, points)
    #    if nodeCount <=201:
    #        for i in range(10):
    #            s = random.randint(0, nodeCount-1)
    #            solution = greedy(s, distance_matrix, points)
    #            solution = three_opt(solution, points, tabu_list)
    #            if value > tour_length(solution, points, nodeCount):
    #                best = solution
    #                value = tour_length(solution, points, nodeCount)
    #        solution = best
    #    else:
    #        s = random.randint(0, nodeCount-1)
    #        solution = greedy(s, distance_matrix, points)
    #        solution = three_opt(solution, points, tabu_list)
    #else:
    #    solution = [i for i in range(nodeCount)]
    #    random.shuffle(solution)
    #    solution = three_opt(solution, points, tabu_list)
    # calculate the length of the tour
    
    #s = random.randint(0, nodeCount-1)
    #solution = greedy(s, distance_matrix, points)
    #solution = three_opt(solution, points, tabu_list)
    solution = [i for i in range(nodeCount)]
    for _ in range(1):
        if nodeCount < 20000:
            s = random.randint(0, nodeCount-1)
            distance_matrix = create_table(nodeCount, points)
            solution = greedy(s, distance_matrix, points)
        else:
            solution = [i for i in range(nodeCount)]
            random.shuffle(solution)
        #print("greedy", tour_length(solution, points, nodeCount))
        #solution = three_opt(solution, points)
        #print("three opt", tour_length(solution, points, nodeCount))
        solution = opt2.two_opt(solution, points)
        #print("two opt", tour_length(solution, points, nodeCount))
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

