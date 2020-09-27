import numpy as np
from scipy import optimize
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

    res_BFGS_linear = optimize.minimize(DLinear, np.array([0,0]), method='BFGS', options={'xatol': EPS, 'disp': True})
    print('BFGS Linear: {}'.format(res_BFGS_linear.x))
    res_BFGS_rational = optimize.minimize(DRational, np.array([0,0]), method='BFGS', options={'xatol': EPS, 'disp': True})
    print('BFGS Rational: {}'.format(res_BFGS_rational.x))


    res_NewtonCG_linear = optimize.minimize(DLinear, np.array([0,0]), method='Newton-CG', jac = jacobian(DLinear), options={'xatol': EPS, 'disp': True})
    print('NewtonCG Linear: {}'.format(res_NewtonCG_linear.x))
    res_NewtonCG_rational = optimize.minimize(DRational, np.array([0,0]), method='Newton-CG', jac = jacobian(DRational), options={'xatol': EPS, 'disp': True})
    print('NewtonCG Rational: {}'.format(res_NewtonCG_rational.x))

    res_CG_linear = optimize.minimize(DLinear, np.array([0,0]), method='CG', tol=EPS)
    print('CG Linear: {}'.format(res_CG_linear.x))
    res_CG_rational = optimize.minimize(DRational, np.array([0,0]), method='CG', tol=EPS)
    print('CG Rational: {}'.format(res_CG_rational.x))

    data_BFGS_linear_y = []
    for k in range(101):
        x_k = k / 100
        data_BFGS_linear_y.append(linearFunction(x_k, res_BFGS_linear.x[0], res_BFGS_linear.x[1]))
    print('Loss for Linear approximation - BFGS: {}'.format(sum(abs(y-x) for x,y in zip(data_y,data_BFGS_linear_y))))

    data_BFGS_rational_y = []
    for k in range(101):
        x_k = k / 100
        data_BFGS_rational_y.append(rationalFunction(x_k, res_BFGS_rational.x[0], res_BFGS_rational.x[1]))
    print('Loss for Rational approximation - BFGS: {}'.format(
        sum(abs(y - x) for x, y in zip(data_y, data_BFGS_rational_y))))

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
    plt.plot(data_x, data_BFGS_linear_y, color='green', label='Linear approximation - BFGS')
    plt.plot(data_x, data_BFGS_rational_y, color='brown', label='Rational approximation - BFGS')
    plt.plot(data_x, data_NewtonCG_linear_y, color='red', label='Linear approximation - NewtonCG')
    plt.plot(data_x, data_NewtonCG_rational_y, color='purple', label='Rational approximation - NewtonCG')
    plt.plot(data_x, data_CG_linear_y, color='black', label='Linear approximation - CG')
    plt.plot(data_x, data_CG_rational_y, color='grey', label='Rational approximation - CG')
    plt.legend(loc="upper left")
    plt.show()



executeTask()