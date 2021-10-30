import GraphILP
from gurobipy import *
import flowConfig as conf
import networkx as nx

class stpModel():
    def __init__(self, G, terminals = [1,2], weight = 'weight', minCapacity = 0, root = 1, pos = None):
        self.G = G
        self.G.G = nx.DiGraph(G.G)
        self.numTerminals = len(terminals)
        self.edges = G.G.edges()
        self.n = G.G.number_of_nodes()
        self.root = root
        for edge in G.G.edges(data=True):
            if ((edge[1], edge[0]) in self.edges):
                continue
            else:
                self.G.G.add_edge(edge[1], edge[0], weight = edge[2]['weight'], capacity = edge[2]['capacity'])
        self.m = Model("PCST Flow Basic Model")
        self.G.set_node_vars(self.m.addVars(G.G.nodes(), vtype=GRB.BINARY))
        self.G.set_edge_vars(self.m.addVars(G.G.edges(), vtype=GRB.BINARY))
        
        self.G.set_label_vars(self.m.addVars(G.G.nodes(), vtype=GRB.INTEGER))
        self.G.flow_variables = self.m.addVars(G.G.edges(), vtype=GRB.INTEGER, lb=0)
        self.m.update()
        self.edges = G.edge_variables
        self.nodes = G.node_variables
        self.labels = G.label_variables
        self.flows = G.flow_variables
        self.edge2var = dict(zip(self.edges.keys(), self.edges.values()))
        self.var2edge = dict(zip(self.edges.values(), self.edges.keys()))
        self.Neighbourhood = []
        self.Capacity = conf.capacity
        self.minCapacity = conf.minCapacity
        for edge in G.G.edges():
            if edge[0] == root:
                self.Neighbourhood.append(edge)
        
        self.addConstraints(weight, terminals)

    def addConstraints(self, weight, terminals):
        # set objective: Maximize the profit
        self.m.setObjective(quicksum([edge_var * self.G.G.edges[edge][weight] for edge, edge_var in self.edges.items()]),
                    GRB.MINIMIZE)

        for e in self.G.G.edges(data=True):
            # Chose at most one direction
            edge = (e[0], e[1])
            reverseEdge = edge[::-1]
            self.m.addConstr(self.edges[edge] + self.edges[reverseEdge] <= 1)
            
            # Maximum Capacity constraint
            self.m.addConstr(self.flows[edge] <= self.edges[edge] * self.Capacity + self.minCapacity)
                    
        # All flow has to start from the Root
        self.m.addConstr(gurobipy.quicksum(self.flows[edge] for edge in self.Neighbourhood if edge[0] == self.root) >= self.numTerminals - 1)   
        
        # No flow to the root is allowed
        for edge in self.Neighbourhood:
            if edge[1] == self.root:
                self.flows[edge] == 0
        
        # Flow conservation constraint
        for j in self.nodes:
            incomingEdges = self.G.G.in_edges(j)
            outgoingEdges = self.G.G.out_edges(j)
            
            # Everything that's not a root or terminal doesn't consume supply
            if j != self.root and j not in terminals:
                self.m.addConstr(sum(self.flows[edgeIn] for edgeIn in incomingEdges) - sum(self.flows[edgeOut] for edgeOut in outgoingEdges) == 0)
            
            # Flow conservation for Terminals. Terminals consume one unit of supply
            elif j != self.root and j in terminals:
                self.m.addConstr(sum(self.flows[edgeIn] for edgeIn in incomingEdges) - sum(self.flows[edgeOut] for edgeOut in outgoingEdges) == conf.terminalConsumption * self.nodes[j])
        
        # Addition by Rolf
        for edge, edge_var in self.edges.items():
            self.m.addConstr(self.numTerminals * edge_var >= self.flows[edge])
            self.m.addConstr(edge_var <= self.flows[edge])

    def extractSolution(self, G):
        r""" Get the optimal Steiner tree in G 
        
            :param G: a weighted ILPGraph
            :param model: a solved Gurobi model for the minimum Steiner tree problem
                
            :return: the edges of an optimal Steiner tree connecting all terminals in G
        """
        solution = []
        for edge, edge_var in G.edge_variables.items():
            if edge_var.X > 0.5:
                solution.append(edge)
        
        return solution