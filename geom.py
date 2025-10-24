from config import *

geom = dde.geometry.Interval(0, L)


def output_transform(x, y):
    return x * (1 - x) * y  # Dirichlet in 0 & L enforced
