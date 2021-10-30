# +
from gurobipy import *
import networkx as nx
import stpModel

class stpModelMtz(stpModel.stpModel):

    def __init__(self, G, terminals = [1,2], weight = 'weight', minCapacity = 0, root = 1, pos = None):
        super().__init__(G, terminals)
        self.addConsts()

    def addConsts(self):
        self.G.set_label_vars(self.m.addVars(self.G.G.nodes(), vtype=GRB.INTEGER, lb=0, ub=self.n))
        self.labels = self.G.label_variables

        self.m.update()  
        edges = self.edges
        labels = self.labels
        n = self.n
        nodes = self.nodes
        m = self.m

        
       # Enforce increasing labels
        for edge, edge_var in self.edges.items():
            self.m.addConstr((n + 1) * edge_var + labels[edge[0]] - labels[edge[1]] <= n )
            
        # Enforce connectivitiy
        for node in self.nodes:
            if node != self.root:
                self.m.addConstr(sum(edges[edge] for edge in self.G.G.in_edges(node)) == nodes[node])
            for edge in self.G.G.out_edges(node):
                self.m.addConstr(edges[edge] <= nodes[node])
        
        

