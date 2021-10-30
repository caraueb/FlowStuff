# +
from graphilp.imports import networkx as imp_nx
import networkx as nx
from gurobipy import *
from datetime import datetime
import pickle
import pandas as pd
# %load_ext autoreload
# %autoreload 2
foundTime = datetime.now()
oldSolution = 0
costsDict = {}
import heuristicConfig as conf


# Find the minimum module required for satisfying the flow on edge (i,j). 
# We only consider three cases and two modules. 
# 0 if no module is needed, i.e. no flow is transported via the edge.
# 1 if a basic transport is needed, i.e., if copper cables suffice.
# 2 If glass fibre is needed.
# Thus we only have to check whether the required Capacity is beyond a certain threshold. If yes, choose fibre, if not either don't use the cable or use the existing copper cable.
def findMinimumModule(G, value, edge):
    if value > G.edges[edge]['capacity']:
        return 2
    elif value == 0:
        return 0
    else:
        return 1

# Computing the weights for each edge
# An edge has a weight > 0 if a higher capacity is needed than the one that already exists on that edge.
# If the maximum capacity of that edge suffices to satisify the flow on that edge, no costs are incurred and thus the edge's weight is 0.
def compEdgeWeights(G, g, b, z):
    global costsDict

    weightGraph = nx.DiGraph()
    
    # Calculating a Dictionary with all the costs of the edges.
    # This has to be done only once. Checking this condition every time the function is called is not good practice but it works.
    if bool(costsDict) == False:
        costsDict = calcCostsDict(G)

    # Compute the edge weight of every edge in the graph.
    for u, v, attr in G.edges(data = True):
        e = (u, v)
        # Find Minimum Module if the demand b is added to the already existing flow on an edge, i.e.,  g[e]. The additional demand on an edge could require a better module to be installed.
        modUp = findMinimumModule(G, g[e] + b, e)
        # Find Minimum Module for the existing demand on the edge.
        modDown = findMinimumModule(G, z[e], e)
        # Find the cost incurred by increasing the edge's capacity. This can also be 0 if the modules are the same in both cases.
        cost = max(0, costsDict[e][modUp] - costsDict[e][modDown])
        # Add the edge to the weight graph.
        weightGraph.add_edge(u, v, weight = cost)
    return weightGraph

# Compute the shortest paths from the root to the node and adds it to the list of shortest Paths.
# This method is called from the Network construction. 
# Parameter node is a terminal with a demand greater than 0.
def createShortestPath(Graph, root, node, weight = None):
    r"""
    :param Graph: A networkx Graph
    :param root: The root of the Graph
    :param node: A terminal node
    :param weight: A string specifying the name of the weight attribute in the Graph.
    """
    sp = nx.shortest_path(Graph, root, node, weight)
    spEdges = []
    
    # Add every node to the Path
    for i in range(len(sp) - 1):
        spEdges.append((sp[i], sp[i+1]))
        
    return spEdges

# Calculates the costs incurred by the flow on an edge.
def calculateCostsFromFlow(z, y, G):
    global costsDict
    r"""
    :param z: An array of size |E| denoting which module is installed on edge (i,j). E.g., z[i][j] = 1 would express that module 1 is installed on edge (i,j)
    :param y: An array of size |E| denoting whether node i is chosen in the current solution. y[i] = 1 would express that node i is included in the solution, i.e. its demand is satisfied 
    :param A networkx Graph.
    """
    
    # Calculate the costs of all edges in the Graph. 
    if bool(costsDict) == False:
        costsDict = calcCostsDict(G)    
    chosenEdges = []
    costs = 0
    # If an edge is active, it incurs costs.
    for key, value in z.items():
        if value > 0:
            costs -= costsDict[key][value]
            chosenEdges.append(key)
    
    # Every satisfied node yields a prize, i.e. profit, for us
    for node, val in y.items():
        if val > 0:
            costs += G.nodes[node]['prize']
    
    return costs, chosenEdges

# Calculate the flow in the Graph
def calculateFlow(G, z, terminals, y, root):
    r"""
    :param z: An array of size |E| denoting which module is installed on edge (i,j). E.g., z[i][j] = 1 would express that module 1 is installed on edge (i,j)
    :param y: An array of size |E| denoting whether node i is chosen in the current solution. y[i] = 1 would express that node i is included in the solution, i.e. its demand is satisfied 
    :param A networkx Graph.
    :param terminals: An array of terminals
    :root: The root of the Graph.
    """
    GC = G.copy()
    totalDemand = 0
    terminalsToPick = terminals.copy()
    flows = dict()
    z_prime = dict()
    
    # Set the demand of the whole graph to 0.
    nx.set_node_attributes(GC, 0, 'demand')

    # Rebuilding the Graph only with edges which are in use, i.e., have some kind of model installed on them and z > 0.
    # Only include the terminals which are included in the Graph. This is done by only adding terminals in the graph that are endpoint of an edge with a z value greater than 1.
    # z is 0 if no module is installed on an edge.
    for key, value in z.items():
        # No module is installed either on the edge.
        if value == 0:
            # Additionaly no module is installed on the reverse edge. This check has to be done since we are working with directed Graphs.
            if z[(key[1], key[0])] == 0:
                GC.remove_edge(key[0], key[1])

        # Additionally add terminals to an extra array if flow is transported to that terminal.
        elif (key[1] in terminalsToPick):
            GC.nodes[key[1]]['demand'] = 1
            terminalsToPick.remove(key[1])
            # The roots capacity is based on totalDemand. Since the root doesn't require inflow but outflow, this value has to be negative.
            totalDemand -= 1
    
    # Temporary saves done for debugging. Not needed in productive code but can be helpful for later stages.
    #nx.write_gpickle(GC, os.path.join(os.getcwd(), 'testInstances', 'tooBig', 'G101.gpickle'))
    #z_pickle = open(os.path.join(os.getcwd(), 'testInstances', 'tooBig', 'z.obj'), 'wb')
    #pickle.dump(z, z_pickle)
    
    # Set the roots demand to the negative amount of terminals. 
    GC.nodes[root]['demand'] = totalDemand

    # Set the edges capacities' based on their z values.
    # The capacity for z = 1 is the base capacity.
    # The capacity fpr z = 2 is the capacity with Glass fibre and thus much higher.
    for edge in GC.edges(data=True):
        acEdge = (edge[0], edge[1])
        if z[acEdge] == 2:
            GC.edges[acEdge]['capacity'] = conf.gfCap
        elif z[acEdge] == 1:
            GC.edges[acEdge]['capacity'] = conf.copperCap
        # Edge's weight computation as stated in 3.2.3 in the paper.
        edge[2]['weight'] = int(( GC.edges[acEdge]['weight'] * z[acEdge]) / GC.edges[acEdge]['capacity'])
    
    # Calculating the min cost flow for the reduced Graph.
    flow = nx.min_cost_flow(GC)
    for node, value in flow.items():
        if value:
            for target, amount in value.items():
                flows[(node, target)] = amount
        else:
            continue

    # Set z_prime depending on the min Cost Flow. z_prime denotes the required module on edge (i,j) for the flow in the min cost flow Graph.
    for edge in G.edges():
        if edge in flows:
            z_prime[edge] = findMinimumModule(G, flows[edge], edge)
        else:
            z_prime[edge] = findMinimumModule(G, 0, edge)
    return z_prime, flows

# Calculates a dictionary of all costs. The costs of each module are arbitrarily chosen by me. In this case, I simply denote the cost of
# module 0 to 0
# module 1 to it's normal cost, i.e., c * 1
# module 2 to double the costs, i.e., c * 2
# The costs are thus only the module index times its costs.
def calcCostsDict(G):
    costsDict = dict()
    for u, v, attr in G.edges(data=True):
        edge = (u, v)
        costsDict[edge] = dict()
        for i in range(3):
            costsDict[edge][i] = attr['weight'] * i
    return costsDict
