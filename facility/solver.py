#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------------------------------------------------------------#
#   
#   Define the MIP model:
#   
#   Y[W,C] - If j warehouse serves i customer <=> Y[j,i] == 1
#   X[W] - If i warehouse is open <=> X[i] == 1
#
#   Distance[W,C] - Distance[i, j] shows the distance between i warehouse and j customer
#
#   Minimize: sum_by_w(facilities[w].setup_cost * X[w]) + sum_all(w,c)(Distance[w, c] * Y[w, c])
#   
#   Subject to:
#           1) sum_by_c(customers[c].demand * Y[w, c]) <= facilities[w].capacity ( for w in warehouses )
#           2) Y[w, c] <= X[w] ( for (w, c) in all_combinations(warehouses, customers) )
#           3) sum_by_w(Y[w, c]) == 1 ( for c in customers )
#
#-----------------------------------------------------------------------------------------------------------------------------------#
#
#       План действий:
#                   1. Составляем матрицу расстояний.       Done!
#                   2. Подготовить данные для MIP модели.      
#-----------------------------------------------------------------------------------------------------------------------------------#
from collections import namedtuple
import math
import random 
import numpy as np
from ortools.linear_solver import pywraplp

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


def countSumDemand(customers):
    res = 0
    for c in customers:
        res += c.demand
    return res

def giveListOfBiggestFacilities(facilities, customers):
    total_demand = countSumDemand(customers)
    sortFacs = sorted(facilities, key = lambda x: x.capacity, reverse=True)
    listToGive = []
    for f in sortFacs:
        listToGive.append(f)
        total_demand -= f.capacity
        if total_demand <= 0:
            break
    return listToGive



def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def nearestCheapFacility(facilities, customers, alpha, beta = 0.2):
    
    solution = [-1]*len(customers)
    capacities = [f.capacity for f in facilities]
    
    for c in customers:
        min_cost = -1
        best_id = -1
        for f in facilities:
            if capacities[f.index] >= c.demand:
                if (min_cost == -1 or (alpha * length(f.location, c.location) + (1 - alpha) * f.setup_cost - beta * capacities[f.index] < min_cost and f.capacity == capacities[f.index])):
                    best_id = f.index 
                    min_cost = alpha * length(f.location, c.location) + (1-alpha) * f.setup_cost - beta * capacities[f.index]
                elif (min_cost == -1 or (alpha * length(f.location, c.location) - beta * capacities[f.index] < min_cost  and f.capacity != capacities[f.index])):
                    best_id = f.index 
                    min_cost = alpha * length(f.location, c.location) - beta * capacities[f.index]
        solution[c.index] = best_id
        capacities[best_id] -= c.demand
    
    used = [0] * len(facilities)
    for facility_index in solution:
        used[facility_index] = 1
    obj = sum([f.setup_cost*used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)

    #obj, solution = opt2(facilities, customers, solution, capacities)


    return obj, solution


def opt2(facilities, customers, solution, capacities):

    quantity_of_customers = [0] * len(facilities)
    for c in solution:
        quantity_of_customers[c] += 1
    for i in range(len(solution)):
        for j in range(i+1, len(solution)):
            f1, c1, f2, c2 = facilities[solution[i]], customers[i], facilities[solution[j]], customers[j]
            f1_c1 = length(f1.location, c1.location)
            f2_c2 = length(f2.location, c2.location)
            f2_c1 = length(f2.location, c1.location)
            f1_c2 = length(f1.location, c2.location)
            d0 = f1_c1 + f2_c2 + f1.setup_cost + f2.setup_cost
            d1 = f2_c1 + f1_c2 + f1.setup_cost + f2.setup_cost
            deltas = [-d0 + d1]
            if (capacities[f1.index] >= c2.demand):
                if quantity_of_customers[f2.index] == 1:
                    d2 = f2_c2 + f2_c1 + f1.setup_cost
                else:
                    d2 = f2_c2 + f2_c1 + f2.setup_cost + f1.setup_cost
                deltas.append(d2 - d0)
            else:
                deltas.append(100000000000000000)

            if (capacities[f2.index] >= c1.demand):
                if quantity_of_customers[f1.index] == 1:
                    d3 = f1_c1 + f1_c2 + f2.setup_cost
                else: 
                    d3 = f1_c1 + f1_c2 + f1.setup_cost + f2.setup_cost
                deltas.append(d3 - d0)
            else:
                deltas.append(100000000000000000)

            min_ = min(deltas)

            if min_ >= 0:
                continue

            temp = None

            if deltas[0] == min_ and capacities[f1.index] + c1.demand >= c2.demand and capacities[f2.index] + c2.demand >= c1.demand:
                temp = solution[i]
                solution[i] = solution[j]
                solution[j] = temp 
                capacities[f1.index] = capacities[f1.index] + c1.demand - c2.demand
                capacities[f2.index] = capacities[f2.index] + c2.demand - c1.demand

            elif deltas[1] == min_ and capacities[f1.index] >= c2.demand:
                solution[c1.index] = solution[c2.index]
                capacities[f2.index] += c2.demand
                capacities[f1.index] -= c2.demand
                quantity_of_customers[f1.index] -= 1

            elif deltas[2] == min_ and capacities[f2.index] >= c1.demand:
                solution[c2.index] = solution[c1.index]
                capacities[f1.index] += c1.demand
                capacities[f2.index] -= c1.demand
                quantity_of_customers[f2.index] -= 1

    used = [0] * len(facilities)
    for facility_index in solution:
        used[facility_index] = 1
    obj = sum([f.setup_cost*used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)
    print(capacities)
    return obj, solution
    

def createData(facilities, customers, facility_count, customer_count):
    data = {}

    #Сначала разбираемся с objective function
    data["num_vars"] = facility_count + customer_count * facility_count
    costs = [facility.setup_cost for facility in facilities]
    distances = []
    for i in range(facility_count):
        for j in range(customer_count):
            distances.append(length(facilities[i].location, customers[j].location))
    data["obj_coefs"] = costs + distances

    #Далее разбираемся с constraints

    #Constraints 1
    #sum_by_c(customers[c].demand * Y[w, c]) <= facilities[w].capacity ( for w in warehouses )
    data["constraint_coefs"] = []
    data["bounds"] = []
    for i, f in enumerate(facilities): 
        #cначала добавляем нули
        line = [0] * facility_count + [0] * facility_count * customer_count
        
        for j in range(customer_count):
            line[facility_count + (customer_count) * (i) + j] = customers[j].demand
        data["bounds"].append(f.capacity)
        data["constraint_coefs"].append(line)

    #Constraint 2
    #Y[w, c] <= X[w] ( for (w, c) in all_combinations(warehouses, customers) )
    # -X[w] + Y[w, c] <= 0 
    
    for i, f in enumerate(facilities):
        line = [0] * facility_count + [0] * facility_count * customer_count
        line[i] = -1
        k = 0
        
        for l in range(customer_count):
            line_copy = line.copy()
            data["constraint_coefs"].append(line_copy)
            data["constraint_coefs"][-1][facility_count + i*customer_count+l] =  1
            data["bounds"].append(0)
                    
    #Constraint 3
    #sum_by_w(Y[w, c]) == 1 ( for c in customers )
    for j, f in enumerate(customers):
        line = [0] * facility_count + [0] * facility_count * customer_count
        for k in range(facility_count + j, len(line),customer_count ):
            line[k] = 1
        data["constraint_coefs"].append(line)
        data["bounds"].append(1)

    data["num_constraints"] = len(data["constraint_coefs"])
    return data


def unmap(mapping, index):
    return mapping.index(index)
    

def solveWithOrtools(facilities, customers, facility_count, customer_count, mapping = None):
    if mapping is None:
        mapping = [f.index for f in facilities] 
    n = facilities[0].index
    facility_count = len(facilities)

    data = createData(facilities, customers, facility_count, customer_count)
    print("jopa")
    solver = pywraplp.Solver.CreateSolver("SCIP")
    x = {}
    for j in range(data['num_vars']):
        x[j] = solver.IntVar(0, 1, 'x[%i]' % j)
    print('Number of variables =', solver.NumVariables())

    for i in range(data['num_constraints']):
        if i >= facility_count + facility_count*customer_count:
            constraint = solver.RowConstraint(data['bounds'][i], data['bounds'][i], '')
        else:
            constraint = solver.RowConstraint(-2, data['bounds'][i], '')
        for j in range(data['num_vars']):
            constraint.SetCoefficient(x[j], data['constraint_coefs'][i][j])
    print('Number of constraints =', solver.NumConstraints())

    objective = solver.Objective()
    for j in range(data['num_vars']):
        objective.SetCoefficient(x[j], data['obj_coefs'][j])
    objective.SetMinimization()

    status = solver.Solve()
    rawSolution = []
    if status == pywraplp.Solver.OPTIMAL:
        print('Objective value =', solver.Objective().Value())
        for j in range(data['num_vars']):
            rawSolution.append(x[j].solution_value())
        print()
        print('Problem solved in %f milliseconds' % solver.wall_time())
        print('Problem solved in %d iterations' % solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
    else:
        print('The problem does not have an optimal solution.')
    X = None
    X = rawSolution[:facility_count]
    rawSolution = rawSolution[facility_count:]
    Y = []
    for i in range(0, len(rawSolution), customer_count):
        Y.append(rawSolution[:customer_count])
        rawSolution = rawSolution[customer_count:]
    Y = np.array(Y)
    Y = Y.T
    solution = [-1]*len(customers)
    for i, y in enumerate(Y):
        for j in range(len(y)):
            if y[j] == 1:
                solution[i] = mapping[j]
    
    used = [0] * facility_count
    for facility_index in solution:
        used[unmap(mapping, facility_index)] = 1
    obj = sum([f.setup_cost*used[unmap(mapping, f.index)] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[unmap(mapping, solution[customer.index])].location)

    return obj, solution

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))
        
    if facility_count * customer_count < 50000:
        obj, solution = solveWithOrtools(facilities, customers, facility_count, customer_count)
    elif facility_count * customer_count < 150000:
        obj, solution = solveWithOrtools(facilities[48:69], customers, facility_count, customer_count)
    else:
        obj, solution = nearestCheapFacility(facilities, customers, 0.85)

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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

