import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

N = 1000
EPS = 0.001
FUNC = lambda x, a, b, c, d: (a * x + b) / (x ** 2 + c * x + d)

def calculate_lse(data, method, a, b, c, d):
    return sum([(method(x, a, b, c, d) - y) ** 2 for (x, y) in data])

def generate_noisy_data():
    tau = np.random.normal(size=N)
    f_k = lambda k: 1 / (k ** 2 - 3 * k + 2)
    xx = []
    yy = []
    for k in range(0, N):
        x_k = 3 * k / N
        y = tau[k]
        if f_k(x_k) < - 100:
            y += -100
        elif -100 <= f_k(x_k) <= 100:
            y += f_k(x_k)
        else:
            y += 100
        xx.append(x_k)
        yy.append(y)
    return list(zip(xx, yy))


def substitude_point(a, b, c, d):
    x = []
    y = []
    for k in range(0, N):
        x_k = 3 * k / N
        x.append(x_k)
        y.append(FUNC(x_k, a, b, c, d))
    return x, y


def build_function(data):
    return lambda a, b, c, d: sum([(FUNC(x, a, b, c, d) - y) ** 2 for (x, y) in data])


def nelder_meald(data):
    f = build_function(data)
    res = optimize.minimize(lambda x: f(x[0], x[1], x[2], x[3]), np.array((0.1, 0.2, 0.3, 0.4)), method='Nelder-Mead')
    print("Nelder-Mead: {}".format(res))
    return res.x


def levenberg_marquardt_method(data):
    f1 = lambda ab: [(FUNC(x_p, ab[0], ab[1], ab[2], ab[3]) - y_p) ** 2 for (x_p, y_p) in data]

    res = optimize.leastsq(f1, np.asarray([0.1, 0.2, 0.3, 0.4]))
    print("Levenberg-Marquardt: {}".format(res))
    return res[0]


def differential_evolution(data):
    f = build_function(data)
    res = optimize.differential_evolution(lambda x: f(x[0], x[1], x[2], x[3]), ((-2, 2), (-2, 2), (-2, 2), (-2, 2)))
    print("Differential evolution: {}".format(res))
    return res.x


def simultaneous_anneal(data):
    f = build_function(data)
    res = optimize.dual_annealing(lambda x: f(x[0], x[1], x[2], x[3]), ((-2, 2), (-2, 2), (-2, 2), (-2, 2)))
    print("Simultaneous anneal: {}".format(res))
    return res.x


def show_data(data, point, name):
    xx = [i for (i, _) in data]
    yy = [i for (_, i) in data]

    plt.scatter(xx, yy, label="initial data", color="red")
    colors = ['green', 'blue', 'yellow','purple','gray']
    idx = 0
    for p, n in zip(point, name):
        xx, yy = substitude_point(*p)
        plt.plot(xx, yy, label=n, color=colors[idx])
        idx += 1
    plt.legend()
    plt.show()


data = generate_noisy_data()
ptr = nelder_meald(data)
print("lse", calculate_lse(data, FUNC, *ptr))
ptr2 = levenberg_marquardt_method(data)
print("lse", calculate_lse(data, FUNC, *ptr2))
ptr3 = differential_evolution(data)
print("lse", calculate_lse(data, FUNC, *ptr3))
ptr4 = simultaneous_anneal(data)
print("lse", calculate_lse(data, FUNC, *ptr4))

show_data(data, [ptr, ptr2, ptr3, ptr4],
          ["Nelder-Mead", "Levenberg-Marquardt", "Differential evolution", "Simultaneous anneal"])
