import numpy as np
import matplotlib.pyplot as plt
import timeit
import functools
from tqdm import tqdm
from functions import constantFunction, sumFunction, productFunction, polynomialFunction, polynomialFunctionHorner, \
    bubbleSort, qsort, timSort
from helpers import createRandomVector
from scipy.optimize import curve_fit

MAX_N = 300
ALGORITHM_ITERATIONS = 5
FUNCTIONS = [constantFunction, sumFunction, productFunction, polynomialFunction, polynomialFunctionHorner, bubbleSort,
             qsort, timSort]

def measureFunctionsPerfomances(functions):
    for function in functions:
        execution_times = np.zeros(MAX_N)
        for n in tqdm(range(1, MAX_N+1), desc=function.__name__):
            v = createRandomVector(n)
            time = timeit.timeit(functools.partial(function, v), number=ALGORITHM_ITERATIONS)
            execution_times[n-1] = time
        plt.plot(range(1, MAX_N+1), execution_times, label=function.__name__)
        plt.legend(loc="upper left")
        plt.show()

measureFunctionsPerfomances(FUNCTIONS)