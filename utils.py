from math import pi

def right(x, L, epsilon=0):
    L_x = [y for y in L if y < x]
    if L_x != []:
        return x-max(L_x)
    else: 
        return x-epsilon*pi

def left(x, L, epsilon=0):
    L_x = [y for y in L if y > x]
    if L_x != []:
        return min(L_x)-x
    else: 
        return (1-epsilon)*pi-x