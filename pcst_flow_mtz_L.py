# +
from gurobipy import *
import networkx as nx
from datetime import datetime

from networkx.algorithms.centrality.betweenness import edge_betweenness
import flowConfig as conf
import pcstModel
time = 0
oldSolution = 0

class pcstModelMtzL(pcstModel.pcstModel):

    def __init__(self, G, terminals = [1,2], weight = 'weight', minCapacity = 0, root = 1, pos = None):
        super().__init__(G, terminals)
        self.addConsts()

    def addConsts(self):
        self.G.set_label_vars(self.m.addVars(self.G.G.nodes(), vtype=GRB.INTEGER, lb=0, ub=self.n))
        self.labels = self.G.label_variables

        edges = self.edges
        labels = self.labels
        n = self.n
        nodes = self.nodes
        m = self.m

        # MTZ L Precalculation
        shortestPaths = nx.shortest_path(self.G.G, self.root)
        spLengths = dict()
        for k, v in shortestPaths.items():
            spLengths[k] = len(v) - 1

        m.update()  
        
        # labeling constraints: enforce increasing labels. MTZ Lifted Reformulation.
        for edge, edge_var in edges.items():
            revEdge = edge[::-1]
            m.addConstr((n + 1 - spLengths[edge[1]]) * edge_var + (n  - 1 - spLengths[edge[1]]) * edges[revEdge] +
                        labels[edge[0]] - labels[edge[1]] 
                        <= n * nodes[edge[0]] - spLengths[edge[1]] * nodes[edge[1]])
        
        # Restricting the Label with upper and lower bounds. MTZ L additional constraint.
        for node in nodes:
            m.addConstr(labels[node] <= n * nodes[node])
            m.addConstr(spLengths[node] * nodes[node] <= labels[node])
