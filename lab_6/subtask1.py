import numpy as np
import random
import time

VERTEX = 100
EDGES = 500
ITERATIONS = 10
EPS = 1e9

random.seed(0)


def createMatrix():
    matrix = np.zeros((VERTEX, VERTEX))
    cnt = 0
    for i in range(VERTEX):
        for j in range(i + 1, VERTEX):
            if cnt < EDGES:
                if random.randint(1, EDGES) >= EDGES / 2:
                    continue
                matrix[i][j] = random.randint(1, EDGES)
                matrix[j][i] = matrix[i][j]
                cnt += 1
    return matrix


def buildList(m):
    array = []
    for i in range(VERTEX):
        array.append([])
        for j in range(VERTEX):
            if m[i][j] != 0:
                array[-1].append((j, int(m[i][j])))
    return array


def djkstra(g, s):
    d = [1e9 for i in range(VERTEX)]
    u = [False for i in range(VERTEX)]
    d[s] = 0
    for i in range(VERTEX):
        v = -1
        for j in range(VERTEX):
            if not u[j] and (v == -1 or d[j] < d[v]):
                v = j
        if d[v] == 1e9:
            break
        u[v] = True
        for j in range(len(g[v])):
            to = g[v][j][0]
            ln = g[v][j][1]
            if d[v] + ln < d[to]:
                d[to] = d[v] + ln
    for i, j in enumerate(d):
        print(i, ' > ', j)
    return d


def fordBellman(g, s):
    d = [1e9 for i in range(VERTEX)]
    d[s] = 0
    e = []
    for i in range(VERTEX):
        for (j, c) in g[i]:
            e.append((i, j, c))
    for i in range(VERTEX):
        for j in range(len(e)):
            if d[e[j][0]] < EPS:
                d[e[j][1]] = min(d[e[j][1]], d[e[j][0]] + e[j][2])
    return d


def executionTime(method, g, v):
    cntr = 0
    for i in range(ITERATIONS):
        start_time = time.monotonic_ns()
        method(g, v)
        end_time = time.monotonic_ns()
        cntr += end_time - start_time
    cntr = cntr / ITERATIONS
    return cntr

m = createMatrix()
l = buildList(m)
r = executionTime(djkstra, l, 2)
print("time djkstra: ", r)
r = executionTime(fordBellman, l, 2)
print("time fb: ", r)
