from functions import FUNCTIONS, CONSTRAINTS
from methods import METHODS, gauss_method
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

EPS = 0.001

#first subtask
def execudeMethods():
    for method in METHODS:
        for function in FUNCTIONS:
            res, f_executions_count, iterations_count = method(function, CONSTRAINTS[function][0], CONSTRAINTS[function][1])
            print('Method {}. Function {}. The result is: {}'.format(method.__name__, function.__name__, res))
            print('Function was executed {} times'.format(f_executions_count))
            print('Iterations count: {}'.format(iterations_count))


execudeMethods()

#second subtask

def subtaskTwo():
    alpha = np.random.rand()
    beta = np.random.rand()
    data_x = []
    data_y = []
    for k in range(101):
        x_k = k / 100
        y_k = alpha * x_k + beta + np.random.rand()
        data_x.append(x_k)
        data_y.append(y_k)

    def linearFunction(x, a, b):
        return a * x + b

    def DLinear(x):
        res = 0
        for k in range(101):
            res += (linearFunction(data_x[k], x[0], x[1]) - data_y[k]) ** 2
        return res

    def rationalFunction(x, a, b):
        return a / (1 + b * x)

    def DRational(x):
        res = 0
        for k in range(101):
            res += (rationalFunction(data_x[k], x[0], x[1]) - data_y[k]) ** 2
        return res

    res_nelder_mead_linear = optimize.minimize(DLinear, np.array([0,0]), method='nelder-mead', options={'xatol': EPS, 'disp': True})
    print('Nelder-Mead Linear: {}'.format(res_nelder_mead_linear.x))
    res_nelder_mead_rational = optimize.minimize(DRational, np.array([0,0]), method='nelder-mead', options={'xatol': EPS, 'disp': True})
    print('Nelder-Mead Rational: {}'.format(res_nelder_mead_rational.x))
    res_brute_linear = optimize.brute(DLinear, [slice(0, 1, 0.01), slice(0, 1, 0.01)],
                 full_output=True, finish=optimize.fmin)
    print('Brute Linear: {}'.format(res_brute_linear[0]))
    res_brute_rational = optimize.brute(DRational, [slice(0, 1, 0.01), slice(0, 1, 0.01)],
                 full_output=True, finish=optimize.fmin)
    print('Brute Linear: {}'.format(res_brute_rational[0]))
    res_gauss_linear = gauss_method(100, 0, 1, linearFunction)
    res_gauss_rational = gauss_method(100, 0, 1, rationalFunction())

    data_nelder_mead_linear_y = []
    for k in range(101):
        x_k = k / 100
        data_nelder_mead_linear_y.append(linearFunction(x_k, res_nelder_mead_linear.x[0], res_nelder_mead_linear.x[1]))
    print('Loss for Linear approximation - Nelder-Mead: {}'.format(sum(abs(y-x) for x,y in zip(data_y,data_nelder_mead_linear_y))))

    data_nelder_mead_rational_y = []
    for k in range(101):
        x_k = k / 100
        data_nelder_mead_rational_y.append(rationalFunction(x_k, res_nelder_mead_rational.x[0], res_nelder_mead_rational.x[1]))
    print('Loss for Rational approximation - Nelder-Mead: {}'.format(
        sum(abs(y - x) for x, y in zip(data_y, data_nelder_mead_rational_y))))

    data_brute_linear_y = []
    for k in range(101):
        x_k = k / 100
        data_brute_linear_y.append(linearFunction(x_k, res_brute_linear[0][0], res_brute_linear[0][1]))
    print('Loss for Linear approximation - Brute: {}'.format(
        sum(abs(y - x) for x, y in zip(data_y, data_brute_linear_y))))

    data_brute_rational_y = []
    for k in range(101):
        x_k = k / 100
        data_brute_rational_y.append(rationalFunction(x_k, res_brute_rational[0][0], res_brute_rational[0][1]))
    print('Loss for Linear approximation - Brute: {}'.format(
        sum(abs(y - x) for x, y in zip(data_y, data_brute_rational_y))))

    data_gauss_linear_y = []
    for k in range(101):
        x_k = k / 100
        data_gauss_linear_y.append(linearFunction(x_k, res_gauss_linear[0][0], res_gauss_linear[0][1]))
    print('Loss for Linear approximation - gauss: {}'.format(
        sum(abs(y - x) for x, y in zip(data_y, data_gauss_linear_y))))

    data_gauss_rational_y = []
    for k in range(101):
        x_k = k / 100
        data_gauss_rational_y.append(rationalFunction(x_k, res_gauss_rational[0][0], res_gauss_rational[0][1]))
    print('Loss for Linear approximation - gauss: {}'.format(
        sum(abs(y - x) for x, y in zip(data_y, data_gauss_rational_y))))

    plt.plot(data_x, data_y, color='blue',  linestyle='--', label='Experimental data')
    plt.plot(data_x, data_nelder_mead_linear_y, color='green', label='Linear approximation - Nelder-Mead')
    plt.plot(data_x, data_nelder_mead_rational_y, color='brown', label='Rational approximation - Nelder-Mead')
    plt.plot(data_x, data_brute_linear_y, color='red', label='Linear approximation - Brute')
    plt.plot(data_x, data_brute_rational_y, color='yellow', label='Rational approximation - Brute')
    plt.plot(data_x, data_gauss_linear_y, color='green', label='Linear approximation - Gauss method')
    plt.plot(data_x, data_gauss_rational_y, color='brown', label='Rational approximation - Gauss method')
    plt.legend(loc="upper left")
    plt.show()



subtaskTwo()