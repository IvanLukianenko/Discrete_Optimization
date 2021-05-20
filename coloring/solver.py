#!/usr/bin/python
# -*- coding: utf-8 -*-
#---------------------------------------------------------------------------------------------------------------------#
#   Variables:
#       color[node_count] - decision variable
#
#   Constraints:
#           (General constraint) 
#           1) for edge in edges:
#               color[edge[0]] != color[edge[1]]
#
#           (Symmetry breakings constraint)
#           2) for node1 in nodes:
#           for node2 in nodes:
#               if node1 < node2:
#                   color[node1] < color[node2]
#
#----------------------------------------------------------------------------------------------------------------------#
from ortools.sat.python import cp_model

def find_max_deg(edges, node_count):
    """
        Вычислим максимальную степень в графе.
    """
    max_deg = 0

    for i in range(0, node_count):
        k = 0
        for edge in edges:
            if i == edge[1] or i == edge[0]:
                k += 1
            if max_deg < k:
                max_deg = k
    
    return max_deg

def solve_it(input_data):

    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
    
    #max deg is the minimum of possible number of colors
    max_deg = find_max_deg(edges, node_count)
    
    i = 0
    status = 0
    while (status != cp_model.OPTIMAL):
        model = cp_model.CpModel()
        color = []
        for node in range(node_count):
            color.append(model.NewIntVar(0, i+int(node_count**2/(float(node_count**2 - 2*edge_count))), "{}".format(node)))
    
        for edge in edges:
            model.Add(color[edge[0]] != color[edge[1]])
    
        solver = cp_model.CpSolver()
    
        solver.parameters.max_time_in_seconds = 10.0

        status = solver.Solve(model)
        i += 1
    solution = []
    
    if status == cp_model.OPTIMAL:
        for c in color:
            solution.append(solver.Value(c))

    count = len(set(solution))
    
    output_data = str(count) + ' ' + str(0) + '\n'
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

