#!/usr/bin/python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------------------------------#
#
#   Model:
#       minimize: 
#                   Sum by i \in V [dist(0, Ti(0)) + sum by <j, k> (dist(j,k)) + dist(Ti(-1),0)]
#       
#       subject to:
#                   1) sum by j \in Ti (dj) <= c 
#                   2) sum by i \in V (j \in Ti) = 1 (j \in N\{0})
#
#
#   Словами:
#           Нужно минимизировать суммарное растояние всех циклов, при этом нужно, чтобы каждый покупатель
#           был обслужен одним грузовиком, и вместимость грузовика хватало на всех.
#
#
#   Дописать локальный поиск так, чтобы удовлетворялись ограничения.
#-------------------------------------------------------------------------------------------------------------#
import math
from collections import namedtuple
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import opt3
import opt2
import numpy as np
Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def tspOnSteroids(tour, customers, vehicle_count):
    depot = customers[0]
    #print("Количество грузовиков: ", vehicle_count)
    #for _ in range(vehicle_count-1):
    #    customers.append(depot)
    tour = opt2.two_opt(tour, customers)

    return tour
    

def findLZI(tour):
    for i in range(len(tour)-1, 0, -1):
        if tour[i].index == 0:
            return i

def findFZI(tour):
    for i in range(0, len(tour)):
        if tour[i].index == 0:
            return i   

def trivalSolution(vehicle_count, vehicle_capacity, depot, customers, customer_count):
    vehicle_tours = []
    customer_count = len(customers)
    remaining_customers = set(customers)
    remaining_customers.remove(depot)

    for v in range(0, vehicle_count):
        # print "Start Vehicle: ",v
        vehicle_tours.append([])
        capacity_remaining = vehicle_capacity
        while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
            used = set()
            order = sorted(remaining_customers, key=lambda customer: -customer.demand*customer_count + customer.index)
            
            for customer in order:
                if capacity_remaining >= customer.demand:
                    capacity_remaining -= customer.demand
                    vehicle_tours[v].append(customer)
                    # print '   add', ci, capacity_remaining
                    used.add(customer)
            remaining_customers -= used
    return vehicle_tours

def calculateObj(vehicle_count, vehicle_tours, depot):
    obj = 0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 0:
            obj += length(depot,vehicle_tour[0])
            for i in range(0, len(vehicle_tour)-1):
                obj += length(vehicle_tour[i],vehicle_tour[i+1])
            obj += length(vehicle_tour[-1],depot)
    return obj

def fromTourToVehicleTours(tour, vehicle_count, n):
    vehicle_tours = [[] for i in range(vehicle_count)]
    
    if tour[0] == 0:
        num_tour = n-1
        for i in range(len(tour)):
            if tour[i].index == 0:
                num_tour += 1
            vehicle_tours[num_tour].append(tour[i])
        
        return vehicle_tours
    else:
        num_tour = n
        lastZeroIndex = findLZI(tour)
        vehicle_tours[num_tour] = tour[lastZeroIndex:] + tour[:findFZI(tour)]
        tour = tour[:lastZeroIndex]
        tour = tour[findFZI(tour):]
        for i in range(len(tour)):
            if tour[i].index == 0:
                num_tour += 1
            vehicle_tours[num_tour].append(tour[i])
        
        return vehicle_tours

def printTour(tour):
    string = ""
    for t in tour:
        string = string + " " + str(t.index)
    print(string)

def sumDemand(vehicle_tours):
    total_demands = []
    for tour in vehicle_tours:
        tot = 0
        for c in tour:
            tot += c.demand
        total_demands.append(tot)
    return total_demands



def rotateCustomers(vehicle_tours, vehicle_capacity):
    total_demands = sumDemand(vehicle_tours)
    for i in range(len(vehicle_tours)):
        actual_demand = 0
        if total_demands[i] > vehicle_capacity:
            for j in range(len(vehicle_tours[i])):
                actual_demand += vehicle_tours[i][j].demand
                if actual_demand > vehicle_capacity:
                    vehicle_tours[(i+1)%len(vehicle_tours)] += vehicle_tours[i][j:]
                    vehicle_tours[i] = vehicle_tours[i][:j]
                    break
    return vehicle_tours

def findNearestDemand(tour, delta):
    demand = [x.demand - delta for x in tour]
    return demand.index(min(demand))



def exchangeCustomers(vehicle_tours, vehicle_capacity, customers):
    total_demands = sumDemand(vehicle_tours)
    deltas = [vehicle_capacity - total_demand for total_demand in total_demands]
    for i in range(len(vehicle_tours)):
        if total_demands[i] > vehicle_capacity:
            customer_index = findNearestDemand(vehicle_tours[i], total_demands[i] - vehicle_capacity)
            for j in range(len(deltas)):
                if deltas[j] + vehicle_tours[i][customer_index].demand > 0:
                    #temp = vehicle_tours[i][customer_index]
                    vehicle_tours[j].append(vehicle_tours[i][customer_index])
                    #vehicle_tours[j].append(customers[0])
                    del vehicle_tours[i][customer_index]
                    #break
    return vehicle_tours

def createDistanceTable(points):
    a = [[0] * (len(points)) for i in range(len(points))]
    a = np.array(a)
    for i in range(len(points)):
        for j in range(len(points)):
            a[i][j] = length(points[i],points[j])
    return a

def create_data_model(vehicle_count, vehicle_capacity, customers):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = createDistanceTable(customers)
    data['num_vehicles'] = vehicle_count
    data['depot'] = 0
    data['demands'] = [x.demand for x in customers]
    data['vehicle_capacities'] = [vehicle_capacity] * vehicle_count
    return data

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    total_distance = 0
    total_load = 0
    vehicle_tours = []
    for vehicle_id in range(data['num_vehicles']):
        vehicle_tours.append([])
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            vehicle_tours[-1].append(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))
    return vehicle_tours

def solveOrTools(vehicle_count, vehicle_capacity, customers):
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(vehicle_count, vehicle_capacity, customers)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        vehicle_tours = print_solution(data, manager, routing, solution)
    else:
        print('No solution found !')
    return vehicle_tours

def solveLikeTspOnSteroids():
    customers_copy = customers.copy()
    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = trivalSolution(vehicle_count, vehicle_capacity, depot, customers_copy, customer_count)

    tourForTsp = []
    tourForTsp += vehicle_tours[0]
    for i in range(1, vehicle_count):
        #tourForTsp += [customers[0]]
        tourForTsp += vehicle_tours[i]
    tour = tourForTsp
    k = vehicle_count
    vehicle_tours_main = []
    while k > 0:
        tour = tspOnSteroids(tour, customers, vehicle_count)
        vehicle_tours_main.append([])
        demand = 0
        i = 0
        print("Остаток тура:")
        printTour(tour)
        while demand < vehicle_capacity:
            
            if len(tour) == 0 or i == len(tour):
                break
            vehicle_tours_main[-1].append(tour[i])

            demand += tour[i].demand
            
            i += 1
        printTour(vehicle_tours_main[-1])  
        print(demand, vehicle_capacity)
        if demand > vehicle_capacity and k != 1:
            vehicle_tours_main[-1].pop()
            i -= 1
        print("Next")
        printTour(vehicle_tours_main[-1])  
        tour = tour[i:]
        k -= 1
        if len(tour) != 0 and k == 0:
            vehicle_tours_main[-1] += tour

        #customer_count -= len(vehicle_tours_main[-1]) +1
    
    vehicle_tours = vehicle_tours_main

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])
    
    customers = []
    for i in range(1, customer_count+1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

    #the depot is always the first customer in the input
    depot = customers[0] 
    vehicle_tours = solveOrTools(vehicle_count, vehicle_capacity, customers)
    print("Принчу туры")
    for i in range(len(vehicle_tours)):
        vehicle_tours[i] = vehicle_tours[i][1:]
        print(" ".join(map(str, vehicle_tours[i])))
    print("Закончил принтить туры")
    # checks that the number of customers served is correct
    assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1
    for i in range(len(vehicle_tours)):
        for j in range(len(vehicle_tours[i])):
            vehicle_tours[i][j] = customers[vehicle_tours[i][j]]
    # calculate the cost of the solution; for each vehicle the length of the route
    obj = calculateObj(vehicle_count, vehicle_tours, depot)
    
    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += str(depot.index) + ' ' + ' '.join([str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'

    return outputData


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

