from collections import defaultdict
class Graph:
    def __init__(self):
        self.Graph = defaultdict(list)

    def addEdge(self, u, v):
        self.Graph[u].append(v)

    def BFS(self,s, goal):
        visited = {i : False for i in self.Graph}
        parent = {i : None for i in self.Graph}
        queue = []
        queue.append(s)
        visited[s] = True
        while queue:
            s = queue.pop(0)
            #print(s, end = ' ')
            if s in goal:
                tg = []
                while s is not None:
                    tg.append(s)
                    s = parent[s]
                tg.reverse()
                print('Đường đi là:', '->'.join(tg))
                return

            for i in self.Graph[s]:
                if i not in visited or visited[i] == False:
                    queue.append(i)
                    visited[i] = True
                    parent[i] = s

g= Graph()
g.addEdge('a', 'c')
g.addEdge('a', 'd')
g.addEdge('a', 'e')
g.addEdge('c', 'f')
g.addEdge('d', 'i')
g.addEdge('d', 'g')
g.addEdge('f','m')
g.addEdge('f','h')
g.addEdge('g','r')
g.addEdge('g','k')
g.BFS('a', ['g', 'h', 'r'])