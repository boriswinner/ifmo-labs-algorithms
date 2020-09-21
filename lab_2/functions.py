import numpy as np

def cubicFunction(x):
    return x**3

def moduleFunction(x):
    return abs(x - 0.2)

def sinFunction(x):
    return x * np.sin(1 / x)

FUNCTIONS = [cubicFunction, moduleFunction, sinFunction]
CONSTRAINTS = {
    cubicFunction: (0, 1),
    moduleFunction: (0, 1),
    sinFunction: (0.01, 1),
}
