from gurobipy import *
import networkx as nx

def create_model(G, terminals = [1,2], weight = 'weight', minCapacity = 0, root = 1, pos = None, prize = 'prize'):    
    r""" Create an ILP for the linear Steiner Problem. 
    
    The model can be seen in Paper Chapter 3.0. This model
    doesn't implement tightened labels.
    
    :param G: an ILPGraph
    :param terminals: a list of nodes that need to be connected by the Steiner tree
    :param weight: name of the argument in the edge dictionary of the graph used to store edge cost
    :return: a Gurobi model
    ILP:
        .. math::
            :nowrap:
    Example:
            .. list-table:: 
               :widths: 50 50
               :header-rows: 0
               * - .. image:: example_steiner.png
                 - `Steiner trees <https://github.com/VF-DE-CDS/GraphILP-API/blob/develop/graphilp/examples/SteinerTreesOnStreetmap.ipynb>`_
                   Find the shortest tree connecting a given set of nodes in a graph.
    """        
    
    # ensure that input is a directed graph
    if type(G.G) != nx.classes.digraph.DiGraph:
        G.G = nx.DiGraph(G.G)
    
    Terminals = len(terminals)
    Capacity = 110
    minCapacity = 0
    edges = G.G.edges()

    # Create Neighbourhood of the Root
    Neighbourhood = []
    for edge in G.G.edges():
        if edge[0] == root:
            Neighbourhood.append(edge)
    
    # create model
    m = Model("Steiner Tree - Flow Formulation")
    
    G.set_node_vars(m.addVars(G.G.nodes(), ub = 1))
    G.set_edge_vars(m.addVars(G.G.edges(), ub = 1))

    G.flow_variables = m.addVars(G.G.edges(), lb=0)
        
    m.update()  

    
    # abbreviations
    edges = G.edge_variables
    nodes = G.node_variables
    flows = G.flow_variables
    edge2var = dict(zip(edges.keys(), edges.values()))
    var2edge = dict(zip(edges.values(), edges.keys()))
    
    # set objective: minimize the sum of the weights of edges selected for the solution
    m.setObjective(quicksum([G.G.nodes[node].get(prize, 0) * node_var for node, node_var in nodes.items()])
                   - quicksum([edge_var * G.G.edges[edge][weight] for edge, edge_var in edges.items()]),
                   GRB.MAXIMIZE)

    for e in G.G.edges(data=True):
        
        # at most one direction per edge can be chosen
        edge = (e[0], e[1])
        reverseEdge = edge[::-1]
        m.addConstr(edges[edge] + edges[reverseEdge] <= 1)
        
        # Maximum Capacity must not be exceeded for every edge
        m.addConstr(flows[edge] <= edges[edge] * Capacity + minCapacity)
                   
    # All flow has to start from the Root
    m.addConstr(gurobipy.quicksum(flows[edge] for edge in Neighbourhood if edge[0] == root) <= Terminals)   
    
    # No flow to the root is allowed
    for edge in Neighbourhood:
        if edge[1] == root:
            flows[edge] == 0
    
    # Flow conservation constraint
    for j in nodes:
        incomingEdges = G.G.in_edges(j)
        outgoingEdges = G.G.out_edges(j)
        
        # Everything that's not a root or terminal doesn't consume supply
        if j != root and j not in terminals:
            m.addConstr(sum(flows[edgeIn] for edgeIn in incomingEdges) - sum(flows[edgeOut] for edgeOut in outgoingEdges) == 0)
        
        # Flow conservation for Terminals. Terminals consume one unit of supply
        # This is a very simple assumption and subject to later change
        elif j != root and j in terminals:
            m.addConstr(sum(flows[edgeIn] for edgeIn in incomingEdges) - sum(flows[edgeOut] for edgeOut in outgoingEdges) == 1)# * nodes[j])
    
    return m, var2edge

# +
def extractSolution(G, model):
    r""" Get the optimal Steiner tree in G 
    
        :param G: a weighted ILPGraph
        :param model: a solved Gurobi model for the minimum Steiner tree problem
            
        :return: the edges of an optimal Steiner tree connecting all terminals in G
    """
    solution = []
    for edge, edge_var in G.edge_variables.items():
        if edge_var.X > 0:
            solution.append(edge)
    return solution

def extractSolutionNodes(G, model):
    r""" Get the optimal Steiner tree in G 
    
        :param G: a weighted ILPGraph
        :param model: a solved Gurobi model for the minimum Steiner tree problem
            
        :return: the edges of an optimal Steiner tree connecting all terminals in G
    """
    solution = []
    for node, node_var in G.node_variables.items():
        if node_var.X > 0:
            solution.append(node)
    return solution

def extractSolutionEdges(G, model):
    solution = []
    #solution = dict()
    for edge, edge_var in G.edge_variables.items():
        if edge_var.X > 0:
            solution.append((edge, edge_var.X, G.G.edges[edge]['weight'], G.G.edges[edge]['capacity']))
            #solution[edge] = edge_var.X
    return solution
    
