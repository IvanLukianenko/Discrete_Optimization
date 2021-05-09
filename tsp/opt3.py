#Владос сорян шо так долго, не забудь заимпортить либы, которые тут юзаются

def reverse_segment_if_3better(tour, i, j, k, t, points):
    """Тоже самое"""
    #Дан путь - [...A-B...C-D...E-F...]
    A, B, C, D, E, F = points[tour[i-1]], points[tour[i]], points[tour[j-1]], points[tour[j]], points[tour[k-1]], points[tour[k % len(tour)]]
    d0 = length(A, B) + length(C, D) + length(E, F)
    d1 = length(A, C) + length(B, D) + length(E, F)
    d2 = length(A, B) + length(C, E) + length(D, F)
    d3 = length(A, D) + length(E, B) + length(C, F)
    d4 = length(F, B) + length(C, D) + length(E, A)

    deltas = [d1 - d0, d2 - d0, d3 - d0, d4 - d0]

    min_ = min(deltas)
    random.random() - вернет вероятность от 0 до 1 
    if min_ >= 0:   
        if random.random() > np.exp(-min_/max(t, 0.0000001)):
            #print("Не Отжигаю")
            return 0
        else:
            #print("Отжигаю")
            pass
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
    """Тоже самое шо и с два оптом"""
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


def all_3segments(n: int):
    """все тройки"""
    return ((i, j, k)
        for i in range(n)
        for j in range(i + 2, n)
        for k in range(j + 2, n + (i > 0)))