import math


def testFunction(x, y):
    """ Retorna la diferencia entera entre dos valores."""
    diff = abs(x - y)
    value = math.floor(diff)
    return value