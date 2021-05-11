import math

def tour_length(solution, points, nodeCount):
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])
    return obj

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def reverse_segment_if_2better_(tour, i, j, points):
    """Чекаем, если уменьшится длина"""
    #[...A-B...C-D...]
    A, B, C, D = points[tour[i-1]], points[tour[i]], points[tour[j-1]], points[tour[j % len(tour)]]
    d0 = length(A, B) + length(C, D)
    d1 = length(A, C) + length(B, D)

    delta = d1 - d0
    if delta >= 0:    
        return 0

    tour[i:j] = reversed(tour[i:j])
    return delta


def two_opt(tour, points):
    """Перебираем все пары и запускаем функцию выше"""
    while True:
        delta = 0
        for (a, b) in all_2segments_(len(tour)):
            delta += reverse_segment_if_2better_(tour, a, b, points)
            #print(tour_length(tour, points, len(points)))
        if delta >= 0:
            break 
    return tour

def all_2segments_(n: int):
    """комбинации всех пар ребер"""
    return ((i, j)
        for i in range(n)
        for j in range(i + 2, n + (i>0)))