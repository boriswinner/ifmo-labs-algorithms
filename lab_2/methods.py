import numpy as np
import math
from scipy import optimize
EPS = 0.001

def bruteForce(func, x_from=0, x_to=1):
    f_min = func(x_from)
    x_min = x_from
    f_executions_count = 1
    iterations_countations_count = 0
    for x in np.arange(x_from + EPS, x_to + EPS, EPS):
        f_executions_count += 1
        iterations_countations_count += 1
        if func(x) <= f_min:
            f_min = func(x)
            x_min = x
    return [x_min, f_executions_count, iterations_countations_count]

def dichotomyMethod(f, x_from, x_to):
    m = EPS / 2
    iterations_countations_count = 0
    f_executions_count = 0
    x_min = x_from
    while abs(x_from - x_to) > EPS:
        iterations_countations_count += 1
        f_executions_count += 2
        x_min = (x_from + x_to - m) / 2
        x2 = (x_from + x_to + m) / 2
        y1 = f(x_min)
        y2 = f(x2)
        if y1 > y2:
            x_from = x_min
        else:
            x_to = x2
    return [x_min, f_executions_count, iterations_countations_count]

def goldenSectionSearch(f, x_from, x_to):
    x1 = x_from + (3 - math.sqrt(5)) / 2 * (x_to - x_from)
    x2 = x_to + (math.sqrt(5) - 3) / 2 * (x_to - x_from)
    y1 = f(x1)
    y2 = f(x2)
    iterations_count = 1
    f_executions_count = 2
    while abs(x_from - x_to) > EPS:
        iterations_count += 1
        f_executions_count += 1
        x1 = x_from + (3 - np.sqrt(5)) / 2 * (x_to - x_from)
        x2 = x_to + (np.sqrt(5) - 3) / 2 * (x_to - x_from)
        if y1 <= y2:
            x_to = x2
            y2 = f(x2)
        else:
            x_from = x1
            y1 = f(x1)
    return [x1, iterations_count, f_executions_count]

def gauss_method(iter_count, a, b, f):
    current_iter = 0
    points = [(a, b)]
    ls = [-1]
    while True:
        cur_a, cur_b = points[-1]
        if len(points) % 2 == 0:
            l = optimize.golden(lambda l1: f(l1, cur_b), brack=(-1, 1))
            next_a = l
            next_b = cur_b
        else:
            l = optimize.golden(lambda l1: f(cur_a, l1), brack=(-1, 1))
            next_a = cur_a
            next_b = l
        ls.append(l)
        points.append((next_a, next_b))
        current_iter += 1
        if abs(f(cur_a, cur_b) - f(next_a, next_b)) < EPS or current_iter > iter_count:
            break
    return points, ls, current_iter

METHODS = [bruteForce, dichotomyMethod, goldenSectionSearch]