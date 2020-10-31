import numpy as np
from scipy import optimize
from sympy import symbols, diff
import matplotlib.pyplot as plt
from autograd import jacobian

EPS = 0.001

def executeTask():
    alpha = np.random.rand()
    beta = np.random.rand()
    data_x = []
    data_y = []
    for k in range(101):
        x_k = k / 100
        y_k = alpha * x_k + beta + np.random.rand()
        data_x.append(x_k)
        data_y.append(y_k)

    def fun_linear(args, x, y):
        return args[0] * x + args[1] - y

    def fun_rational(args, x, y):
        return args[0] / (1 + args[1] * x) - y

    def linearFunction(x, a, b):
        return a * x + b

    def rationalFunction(x, a, b):
        return a / (1 + b * x)

    def gradient_descent(a_init, b_init, learning_rate, cost_func):
        a_curr = a_init
        b_curr = b_init
        curr_coeffs = (a_curr, b_curr)
        cost_start = cost_func(curr_coeffs)
        iterations = 0
        fCalc = 1
        running = True

        while running:
            iterations += 1
            fCalc += 1
            curr_coeffs = symbols('a b', real=True)
            a, b = curr_coeffs
            d_a = diff(cost_func(curr_coeffs), a)
            d_a = float(d_a.subs({a: a_curr, b: b_curr}))
            d_b = diff(cost_func(curr_coeffs), b)
            d_b = float(d_b.subs({a: a_curr, b: b_curr}))

            a_curr = a_curr - learning_rate * d_a
            b_curr = b_curr - learning_rate * d_b
            curr_coeffs = (a_curr, b_curr)

            cost_curr = cost_func(curr_coeffs)

            if abs(cost_curr - cost_start) < EPS:
                running = False
            else:
                cost_start = cost_curr

        return [a_curr, b_curr, cost_func(curr_coeffs), iterations, fCalc]

    def lin_cost_function(x):
        a, b = x
        res = 0
        for k in range(101):
            res += (linearFunction(data_x[k], a, b) - data_y[k]) ** 2
        return res

    def rat_cost_function(x):
        a, b = x
        res = 0
        for k in range(101):
            res += (rationalFunction(data_x[k], a, b) - data_y[k]) ** 2
        return res

    def DLinear(x):
        res = 0
        for k in range(101):
            res += (linearFunction(data_x[k], x[0], x[1]) - data_y[k]) ** 2
        return res

    def DRational(x):
        res = 0
        for k in range(101):
            res += (rationalFunction(data_x[k], x[0], x[1]) - data_y[k]) ** 2
        return res

    a, b, cost_curr, iterations, fCalc = gradient_descent(a_init=0, b_init=0, learning_rate=0.001,
                                                          cost_func=lin_cost_function)
    res_GD_linear = []
    res_GD_linear.append(a)
    res_GD_linear.append(b)
    print('Gradient descent Linear: {}'.format(res_GD_linear))

    a, b, cost_curr, iterations, fCalc = gradient_descent(a_init=0, b_init=0, learning_rate=0.001,
                                                          cost_func=rat_cost_function)
    res_GD_rational = []
    res_GD_rational.append(a)
    res_GD_rational.append(b)
    print('Gradient descent rational: {}'.format(res_GD_rational))

    res_NewtonCG_linear = optimize.minimize(DLinear, np.array([0,0]), method='Newton-CG', jac = jacobian(DLinear), options={'xatol': EPS, 'disp': True})
    print('NewtonCG Linear: {}'.format(res_NewtonCG_linear.x))
    res_NewtonCG_rational = optimize.minimize(DRational, np.array([0,0]), method='Newton-CG', jac = jacobian(DRational), options={'xatol': EPS, 'disp': True})
    print('NewtonCG Rational: {}'.format(res_NewtonCG_rational.x))

    res_CG_linear = optimize.minimize(DLinear, np.array([0,0]), method='CG', tol=EPS)
    print('CG Linear: {}'.format(res_CG_linear.x))
    res_CG_rational = optimize.minimize(DRational, np.array([0,0]), method='CG', tol=EPS)
    print('CG Rational: {}'.format(res_CG_rational.x))

    res_LMA_linear = optimize.least_squares(fun_linear, np.array([0, 0]), args=(data_x, data_y), method='lm', gtol=0.001, )
    print('LMA Linear: {}'.format(res_LMA_linear.x))

    res_LMA_rational = optimize.least_squares(fun_linear, np.array([0, 0]), args=(data_x, data_y), method='lm', gtol=0.001, )
    print('LMA Rational: {}'.format(res_LMA_rational.x))

    data_GD_linear_y = []
    for k in range(101):
        x_k = k / 100
        data_GD_linear_y.append(linearFunction(x_k, res_GD_linear[0], res_GD_linear[1]))
    print('Loss for Linear approximation - Gradient Descent: {}'.format(sum(abs(y-x) for x,y in zip(data_y,data_GD_linear_y))))

    data_GD_rational_y = []
    for k in range(101):
        x_k = k / 100
        data_GD_rational_y.append(rationalFunction(x_k, res_GD_rational[0], res_GD_rational[1]))
    print('Loss for rational approximation - Gradient Descent: {}'.format(sum(abs(y-x) for x,y in zip(data_y,data_GD_rational_y))))

    data_NewtonCG_linear_y = []
    for k in range(101):
        x_k = k / 100
        data_NewtonCG_linear_y.append(linearFunction(x_k, res_NewtonCG_linear.x[0], res_NewtonCG_linear.x[1]))
    print('Loss for Linear approximation - NewtonCG: {}'.format(sum(abs(y-x) for x,y in zip(data_y,data_NewtonCG_linear_y))))

    data_NewtonCG_rational_y = []
    for k in range(101):
        x_k = k / 100
        data_NewtonCG_rational_y.append(rationalFunction(x_k, res_NewtonCG_rational.x[0], res_NewtonCG_rational.x[1]))
    print('Loss for Rational approximation - NewtonCG: {}'.format(
        sum(abs(y - x) for x, y in zip(data_y, data_NewtonCG_rational_y))))

    data_LMA_linear_y = []
    for k in range(101):
        x_k = k / 100
        data_LMA_linear_y.append(linearFunction(x_k, res_LMA_linear.x[0], res_LMA_linear.x[1]))
    print('Loss for Linear approximation - LMA: {}'.format(sum(abs(y-x) for x,y in zip(data_y,data_LMA_linear_y))))

    data_LMA_rational_y = []
    for k in range(101):
        x_k = k / 100
        data_LMA_rational_y.append(rationalFunction(x_k, res_LMA_rational.x[0], res_LMA_rational.x[1]))
    print('Loss for Rational approximation - LMA: {}'.format(
        sum(abs(y - x) for x, y in zip(data_y, data_LMA_rational_y))))


    data_CG_linear_y = []
    for k in range(101):
        x_k = k / 100
        data_CG_linear_y.append(linearFunction(x_k, res_CG_linear.x[0], res_CG_linear.x[1]))
    print('Loss for Linear approximation - CG: {}'.format(sum(abs(y-x) for x,y in zip(data_y,data_CG_linear_y))))

    data_CG_rational_y = []
    for k in range(101):
        x_k = k / 100
        data_CG_rational_y.append(rationalFunction(x_k, res_CG_rational.x[0], res_CG_rational.x[1]))
    print('Loss for Rational approximation - CG: {}'.format(
        sum(abs(y - x) for x, y in zip(data_y, data_CG_rational_y))))

    plt.plot(data_x, data_y, color='blue',  linestyle='--', label='Experimental data')
    plt.plot(data_x, data_GD_linear_y, color='green', label='Linear approximation - Gradient descent')
    plt.plot(data_x, data_GD_rational_y, color='brown', label='Rational approximation - Gradient descent')
    plt.plot(data_x, data_NewtonCG_linear_y, color='red', label='Linear approximation - NewtonCG')
    plt.plot(data_x, data_NewtonCG_rational_y, color='purple', label='Rational approximation - NewtonCG')
    plt.plot(data_x, data_CG_linear_y, color='black', label='Linear approximation - CG')
    plt.plot(data_x, data_CG_rational_y, color='grey', label='Rational approximation - CG')
    plt.plot(data_x, data_LMA_linear_y, color='red', label='Linear approximation - LMA')
    plt.plot(data_x, data_LMA_rational_y, color='purple', label='Rational approximation - LMA')
    plt.legend(loc="upper left")
    plt.show()


executeTask()