import queue
from constants import EDGES, VERTEXES

def dfs(v: int, color: int, l: list, color_list: list):
    color_list[v] = color
    for to in l[v]:
        if color_list[to] == 0:
            dfs(to, color, l, color_list)

def dfs_init(l):
    color_list = [0 for _ in range(VERTEXES)]
    color = 1
    for i in range(VERTEXES):
        if color_list[i] == 0:
            dfs(i, color, l, color_list)
            color += 1
    color = color - 1
    return color


def bfs(l: list, v: int):
    dist = [101 for _ in range(VERTEXES)]
    q = queue.Queue()
    q.put(v)
    dist[v] = 0
    while not q.empty():
        v = q.get()
        for to in l[v]:
            if dist[to] == 101:
                dist[to] = dist[v] + 1
                q.put(to)
    for i in range(VERTEXES):
        print(str(i).zfill(2) + ">" + str(dist[i]).zfill(3))
    return dist