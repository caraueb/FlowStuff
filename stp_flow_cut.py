# +
from gurobipy import *
import networkx as nx
import matplotlib.pyplot as plt

var2edge = None
terms = None
var2node = None
node2var = None
x = None
terms = None
Graph = None
edge2var = None
edge2attr = None
edgs = None
rootNode = None
positions = None

def create_model(G, terminals = [1,2], weight = 'weight', minCapacity = 0, root = 1, pos = None):    
    global rootNode,var2edge, terms,var2node, x, terms, Graph, edge2var, edgs, edge2attr, positions, node2var
    
    # ensure that input is a directed graph
    if type(G.G) != nx.classes.digraph.DiGraph:
        G.G = nx.DiGraph(G.G)
    Graph = G.G
    positions = pos
    Terminals = len(terminals)
    rootNode = root
    #minCapacity = 64
    
    edges = G.G.edges()
    terms = set(terminals)
    # Get Outflow edges from Root
    Neighbourhood = []
    for edge in G.G.edges():
        if edge[0] == root:
            Neighbourhood.append(edge)
    
    # create model
    m = Model("Steiner Tree - Flow Formulation")
    m.Params.LazyConstraints = 1
    
    # create reverse edge for every edge in the graph
    for edge in G.G.edges(data=True):
        if ((edge[1], edge[0]) in edges):
            continue
        else:
            G.G.add_edge(edge[1], edge[0], weight = edge[2][weight])
    
    edgs = G.G.edges(data=True)

    # Edge variables y
    y = {}
    # Node variable x
    x = {}
    
    for edge in G.G.edges():
        y[edge] = m.addVar(vtype = gurobipy.GRB.BINARY, name = "y" + str(edge))
    for node in G.G.nodes():
        x[node] = m.addVar(vtype = gurobipy.GRB.BINARY, name = "x" + str(node))
        
    m.update() 
    
    var2edge = dict(zip(y.values(), y.keys()))
    var2node = dict(zip(x.values(), x.keys()))
    
    edge2var = dict(zip(y.keys(), y.values()))
    edge2attr = dict()
    for edge in edgs:
        edge2attr[edge[0], edge[1]] = edge[2]
    node2var = dict(zip(x.keys(), x.values()))
    
    # abbreviations
    edges = G.G.edges()
    nodes = G.G.nodes()
    # set objective: minimize the sum of the weights of edges selected for the solution
    m.setObjective(gurobipy.quicksum([y[edge[0],edge[1]] * edge[2][weight] for edge in G.G.edges(data=True)]), GRB.MINIMIZE)

    for node in nodes:
        # the outer loop makes sure that terminals that are not in the graph are ignored
        # Every Terminal must be included in the solution
        if node in terminals or node == root:
            m.addConstr(x[node] == 1)
    
    # prohibit isolated vertices
    for node in nodes:
        m.addConstr(x[node] - sum(y[edge] for edge in edges if (node == edge[0]) or (node == edge[1])) <= 0)
        #inEdges = G.G.in_edges(node)
        #m.addConstr(sum(y[edge] for edge in inEdges) >= x[node])
    
    for e in G.G.edges(data=True):
        
        # at most one direction per edge can be chosen
        # runtime can probably be greatly improved if iterating is done in a smarter way
        edge = (e[0], e[1])
        reverseEdge = edge[::-1]
        m.addConstr(y[edge] + y[reverseEdge] <= x[e[0]])

    return m, var2edge

def extractSolution(G, model):
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

def cut_cycle(model, where):
    global x, terms, rootNode, edgs, edge2attr, positions, node2var

    if where == GRB.Callback.MIPSOL or where == GRB.Callback.MIPNODE:
        variables = model.getVars()
        if where == GRB.Callback.MIPNODE:
            
            cur_sol = model.cbGetNodeRel(variables)
        else:
            cur_sol = model.cbGetSolution(variables)
            
        solNodes = dict()
        solEdges = []
        solutionNodes = []
        for i, var in enumerate(variables):
            if var in var2edge and cur_sol[i] > 0:
                solEdges.append((var2edge[var][0], var2edge[var][1], {'value':cur_sol[i]}))
            if var in var2node and cur_sol[i] > 0:
                solNodes[var2node[var]] = cur_sol[i]
                solutionNodes.append(var2node[var])
                    
        # Fill the updated Capacities Dictionary in order to update G' capacities
        updatedCapacities = dict()
        for edge in solEdges:
            edge[2]['capacity'] = edge2attr[(edge[0], edge[1])]['capacity'] * edge[2]['value']
        
        G2 = nx.DiGraph()
        G3 = Graph.copy()
        G2.add_edges_from(solEdges)
        
        # Add the artificial sink to G'
        G2.add_node(0)
        G2.add_node(rootNode)
        
        # Find the terminals that were chosen in the solution
        termsInSolution = terms.intersection(solutionNodes)
        for t in termsInSolution:
            # Add edge from terminals to sink with capacity of 1
            G2.add_edge(t, 0, capacity = 1)
            
        # Calculate min cut max flow
        maxFlow, sets = nx.minimum_cut(G2, rootNode, 0)
        
        rootSet, terminalSet = sets
        
        # Find inflowing edges into the terminalSet. This includes edges between Nodes which are both in the Set
        inEdges = G3.in_edges(terminalSet)
        
        # Above mentioned Edges need to be removed
        edgesToRemove = []
        for edge in G3.edges():
            # Edges that connect nodes that are both in the terminal Set are found
            if (edge[0] in terminalSet and edge[1] in terminalSet):
                # and added to the array which later defines the edges that need to be removed from the Graph
                edgesToRemove.append(edge)
        
        G3.remove_edges_from(edgesToRemove)
        
        # Find the Terminal Nodes in the minCut defining Set including the Sink
        termsInSecond = terminalSet.intersection(termsInSolution)
        totalDemand = 0
        for node in terms:
            totalDemand += solNodes.get(node, 0)
        
        if maxFlow < totalDemand:
            # Recalculate all edges that lead into the terminal Set. inEdges now doesn't include edges inside the Set anymore
            inEdges = G3.in_edges(terminalSet, True)
            
            if len(inEdges) != 0:
                # Save a plot of the Graph
                model.cbLazy(sum(edge2var[edge[0], edge[1]] * edge[2]['capacity'] for edge in inEdges) >= 
                             sum(node2var[node] * 1 for node in termsInSecond))
# -


