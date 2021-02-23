
global_max = 0
global n
n = 3
w = [5, 8, 3]
v = [45, 48, 35]
K = 10
tkn = []
def LS(o_v, f_s, actl_v, tkn):
    global global_max
    global tkn1
    if (f_s <= 0):
        return 0, []
    if (actl_v < global_max):
        return 0, []
    if (len(tkn) == n) and (actl_v< global_max):
        return 0, []
    else:
        if (len(tkn) == n):
            tkn1 = tkn
            return actl_v, tkn
    if (o_v < global_max):
        return 0, []

    a, b = LS(o_v, f_s - w[len(tkn)], actl_v + v[len(tkn)], tkn + [1])
    a1, b1 = LS(o_v - v[len(tkn)], f_s, actl_v, tkn + [0])

    if a > a1:
        global_max, tkn = a, b
    else:
        global_max, tkn = a1, b1
    return global_max, tkn

value, taken = LS(128, K, 0, [])
print(value)
print(taken)
