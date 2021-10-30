# +
from gurobipy import *
import networkx as nx
import stpModel

class stpModelMtzLPlus(stpModel.stpModel):
    
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
        root = self.root

        # MTZ L Precalculation
        shortestPaths = nx.shortest_path(self.G.G, self.root)
        spLengths = dict()
        for k, v in shortestPaths.items():
            spLengths[k] = len(v) - 1

        m.update()  
        
        # labeling constraints: enforce increasing labels. MTZ Lifted Reformulation
        for edge, edge_var in edges.items():
            revEdge = edge[::-1]
            m.addConstr((n + 1 - spLengths[edge[1]]) * edge_var + (n  - 1 - spLengths[edge[1]]) * edges[revEdge] +
                        labels[edge[0]] - labels[edge[1]] 
                        <= n * nodes[edge[0]] - spLengths[edge[1]] * nodes[edge[1]])
            
        # Restricting the Label with upper and lower bounds. Lift from Constraint (11)
        for node in nodes:
            contained = False
            # MTZ L Plus Lifts
            inEdges = self.G.G.in_edges(node)
            outEdges = self.G.G.out_edges(node)
            for inEdge in inEdges:
                if root == inEdge[0]:
                    contained = True
                    break
            # This constraint somehow breaks the model. Im not including it since it doesn't really provide any kind of 
            # tightening anyway
            if contained:
                m.addConstr(labels[node] <= n * nodes[node] - (n - 1) * edges[(root, node)])
            else:
                m.addConstr(
                    spLengths[node] * nodes[node] 
                    + sum((spLengths[edge[0]] - spLengths[node] + 1) * edges[edge] for edge in inEdges) 
                    <= labels[node])
                m.addConstr(labels[node] <= n * nodes[node] - sum(edges[edge] for edge in outEdges))
        
        # Restricting the Label with upper and lower bounds
        for node in nodes:
            m.addConstr(labels[node] <= n * nodes[node])
            m.addConstr(spLengths[node] * nodes[node] <= labels[node])
