#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])
Item_2 = namedtuple("Item", ['index', 'value', 'weight', 'value_on_weight'])
global_max = -1

def BranchAndBound(o_v, f_s, actl_v, tkn):
    """
        Метод ветвей и границ для рюкзака.
    """
    global global_max
    if (f_s < 0):
        return -1, []
    if (len(tkn) == n) and (actl_v < global_max):
        return -1, []
    else:
        if (len(tkn) == n):
            return actl_v, tkn
    if (o_v < global_max):
        return -1, tkn
    if (o_v < 0):
        return -1, tkn
    a, b = BranchAndBound(o_v, f_s - items_2_sorted[len(tkn)].weight, actl_v + items_2_sorted[len(tkn)].value, tkn + [items_2_sorted[len(tkn)].index])
    a1, b1 = BranchAndBound(o_v - items_2_sorted[len(tkn)].value, f_s, actl_v, tkn + [items_2_sorted[len(tkn)].index])

    if a > a1:
        global_max, tkn = a, b
    else:
        global_max, tkn = a1, b1
    return global_max, tkn

def Traceback(arr, K, items_):
    """
        По матрице вычислим решение.
    """
    items = []
    j = len(items_)
    i = K
    while (arr[i][j] != 0):
        if(arr[i][j] == arr[i][j-1]):
            j-=1
            
        else:
            items.append(items_[j-1].index)
            j-=1
            i-=items_[j].weight
            #print("here")
            
    return items

def create_table(items, K):
    """
        Создаем "ту самую матрицу" для динамического программирования рюкзака.
    """
    a = [[0] * (len(items)+1) for i in range(K+1)]
    #print (w)
    for i in range(K+1):
        for j in range(len(items)+1):
            if j == 0:
                a[i][j] = 0
            else:
                if(items[j-1].weight <= i):
                    a[i][j] = max(a[i][j-1], items[j-1].value + a[i - items[j-1].weight][j-1])
                else:
                    a[i][j] = a[i][j-1]
    return a


def greedy_algo(items_2, K):
    """
        Жадный алгоритм.
    """
    actual_weight = 0
    actual_value = 0
    taken = []
    items_2 = sorted(items_2, key = lambda item: item.value_on_weight, reverse = True)  
    i = 0
    while(actual_weight <= K):
        actual_value += items_2[i].value
        actual_weight += items_2[i].weight
        taken.append(items_2[i].index)
        i += 1

    del taken[-1]
    return taken, actual_value - items_2[i-1].value
    

def solve_it(input_data):

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
        items_2.append(Item_2(i-1, int(parts[0]), int(parts[1]), (float(parts[0])/int(parts[1])) ) ) 
    
    taken = [0]*len(items)

    K = capacity
    items_2_sorted = sorted(items_2, key = lambda item: item.value_on_weight, reverse = True) 
    
    if item_count == 400:
        n = 30
        table = create_table(items_2_sorted[:n],K)
        value = table[-1][-1]
        taken_1 = Traceback(table, K, items_2_sorted[:n])
        for i in taken_1:
            taken[i] = 1
    else:
        if len(items) == 10000:
            n = 30
            table = create_table(items_2_sorted[:n],K)
            value = table[-1][-1]
            taken_1 = Traceback(table, K, items_2_sorted[:n])
            for i in taken_1:
                taken[i] = 1
        else:
            table = create_table(items_2_sorted,K)
            value = table[-1][-1]
            taken_1 = Traceback(table, K, items_2_sorted)
            for i in taken_1:
                taken[i] = 1
    
    
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