#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])
Item_2 = namedtuple("Item", ['index', 'value', 'weight', 'value_on_weight'])
def Traceback(arr, K, w):
    items = []
    j = len(w)
    i = K
    while (arr[i][j] != 0):
        if(arr[i][j] == arr[i][j-1]):
            j-=1
            
        else:
            items.append(j)
            j-=1
            i-=w[j]
            #print("here")
            
    return items

def create_table(v, w, K):
    a = [[0] * (len(v)+1) for i in range(K+1)]
    #print (w)
    for i in range(K+1):
        for j in range(len(v)+1):
            if j == 0:
                a[i][j] = 0
            else:
                if(w[j-1] <= i):
                    a[i][j] = max(a[i][j-1], v[j-1] + a[i - w[j-1]][j-1])
                else:
                    a[i][j] = a[i][j-1]
    return a


def greedy_algo(items_2, K):
    actual_weight = 0
    actual_value = 0
    taken = []
    items_2 = sorted(items_2, key = lambda item: item.value_on_weight, reverse = True)  
    i = 0
    for i in range(len(items_2)):
        if actual_weight <= K:
            actual_value += items_2[i].value
            actual_weight += items_2[i].weight
            taken.append(items_2[i].index)
    del taken[-1]
    return taken, actual_value - items_2[i].value






def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []
    items_2 = []
    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
        items_2.append(Item_2(i-1, int(parts[0]), int(parts[1]), int(parts[0])/int(parts[1]) ) ) 


    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    
    #value = 0
    #weight = 0
    taken = [0]*len(items)

    v = [item.value for item in items]
    w = [item.weight for item in items]
    K = capacity
    #print(w)
    if len(items)*K < 1e9:
        table = create_table(v,w,K)
        value = table[-1][-1]
        taken_1 = Traceback(table, K, w)
        for i in taken_1:
            taken[i-1] = 1
    else: 
        taken_2, value= greedy_algo(items_2, K)
        for i in taken_2:
            taken[i-1] = 1
    #print(table)
    #for item in items:
    #    if weight + item.weight <= capacity:
    #        taken[item.index] = 1
    #        value += item.value
    #        weight += item.weight
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

