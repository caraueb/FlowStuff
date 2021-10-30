# +
import networkx as nx
from gurobipy import *
from datetime import datetime
from pcstHeuristicModule import *
import pandas as pd
import numpy as np
import time
# %load_ext autoreload
# %autoreload 2
foundTime = datetime.now()
oldSolution = 0
    
def find_capacity_from_z(z):
    if z < 1:
        return 0
    elif z == 1:
        return 102
    elif z == 2:
        return 1000

def findMinimumModule(pd_edges, colToTake = 'requiredCap', b = 0, colToSave = 'z'):
    # Finding minimum module if g[e] + b is taken as input
    conditions = [pd_edges[colToTake] + b > pd_edges['capacity'],
                    ((pd_edges[colToTake] + b<= pd_edges['capacity']) & (pd_edges[colToTake] + b > 0.0000001)),
                    pd_edges[colToTake] < 0.0000001]
    choices = [2, 1, 0]
    pd_edges[colToSave] = np.select(conditions, choices)
    return pd_edges

def compEdgeWeights(G, pd_edges, b, z_to_take, g_to_take):
    # Making new DiGraph from the Input Graph
    weightGraph = nx.DiGraph(G)    
    
    findMinimumModule(pd_edges, g_to_take, b, 'modUp')

    # Calculating the weight if demand b is taken into account for the choice on the module
    pd_edges['costsModUp'] = pd_edges['modUp'] * pd_edges['weight']
    
    # Calculating the weight if demand b is not taken into account for the choice on the module
    pd_edges['costsNormal'] = pd_edges[z_to_take] * pd_edges['weight']
    
    # Calculating the cost of every edge. Costs only exist if an upgrade needs to be made due to demand b
    pd_edges['cost'] = pd_edges['costsModUp'] - pd_edges['costsNormal']
    
    # Erasing all negative values by setting it to 0. Same as max(0, pd_edges['cost'])
    pd_edges.loc[pd_edges['cost'] < 0, 'cost'] = 0
    
    # Transforming the costs of each edge to an edgeList
    edgeList = pd_edges[['source', 'target','cost']].values.tolist()
    
    # Add the weight of each edge to the DiGraph
    weightGraph.add_weighted_edges_from(edgeList, weight='cost')
    return weightGraph

def createShortestPath(Graph, root, node, weight = None):
    # Compute the shortest paths from the root to every node whose demand is higher than 0.
    # Only solution Nodes have a demand higher than 0.
    sp = nx.shortest_path(Graph, root, node, weight)
    spEdges = []
    
    # Add every node to the Path
    for i in range(len(sp) - 1):
        spEdges.append((sp[i], sp[i+1]))
        
    return spEdges

def calculateCostsFromFlow(z, y, G):
    costsDict = calcCostsDict(G)    
    chosenEdges = []
    costs = 0
    for key, value in z.items():
        if value > 0:
            costs -= costsDict[key][value]
            chosenEdges.append(key)
        
    for node, val in y.items():
        if val > 0:
            costs += G.nodes[node]['prize']
    
    return costs, chosenEdges

def calculateFlow(G, pd_edges, pd_nodes, terminals, root):
    edge_amount = G.number_of_edges()
    flows = dict()
    totalDemand = 0

    pd_nodes = pd_nodes.assign(demand = 0)
    # Create Subset of DataFrame. Only include edges where terminals are either source or target and a flow exists (z > 0)
    pd_sub = pd_edges[(pd_edges['target'].isin(terminals)) & pd_edges['z'] > 0].copy()
    
    # Get the unique targets of all existing Flows
    uniqueTargetsPandas = pd_sub[pd_sub['target'].isin(terminals)].drop_duplicates(subset = ['target'])
    
    # Transform the unique targets to a list of Terminals
    uniqueTargets = uniqueTargetsPandas['target'].values.tolist()
    
    # Get the total demand, i.e. the amount of terminals that receive a flow
    totalDemand = -uniqueTargetsPandas.shape[0]
    
    
    # Find all edges with a module present and put it into the list of active edges.
    activeEdges = [tuple(r) for r in pd_edges.loc[pd_edges['z'] > 0, ['source','target']].values.tolist()]
    
    revEdges = []
    for edge in activeEdges:
        revEdges.append((edge[1], edge[0]))
    activeEdges.extend(revEdges)
    found = set()
    keep = []

    for item in activeEdges:
        if item not in found:
            found.add(item)
            keep.append(item)
    
    # Dataframe for the edges which have a module installed
    activeEdgesDf = pd_edges[pd_edges.index.isin(keep)].copy()
    edgeList = pd_edges[pd_edges['z'] > 0].copy()
    edgeList = edgeList.append(activeEdgesDf)
    
    pd_edges.assign(z_prime = np.zeros(edge_amount), flow = np.zeros(edge_amount))
    if (edgeList.empty == False):
        # Transform all those edges into a list in order to transform it to a Graph
        edges = edgeList[['source','target']].values.tolist()
        GC = nx.DiGraph()
        
        # Add the Edges to the Graph
        GC.add_edges_from(edges)
        nx.set_node_attributes(GC, 0, 'demand')
        for node in uniqueTargets:
            GC.nodes[node]['demand'] = 1
        GC.nodes[root]['demand'] = int(totalDemand)
        
        # Find the capacity of each edge which solely depends on the z value of an edge.
        funcFindCap = np.vectorize(find_capacity_from_z)

        edgeList['flowCapacity'] = funcFindCap(edgeList['z'])
        edgeList['flowWeight'] = (edgeList['weight'] * edgeList['z']) / edgeList['capacity']
        edgeList['flowWeight'] = edgeList['flowWeight'].astype(int)
        
        # Extract attribute information from the Pandas Dataframe
        attrDict = edgeList[['flowWeight','flowCapacity']].to_dict()
        
        # Add the Attributes to the Graph
        nx.set_edge_attributes(GC, attrDict['flowWeight'], name = 'weight')
        nx.set_edge_attributes(GC, attrDict['flowCapacity'], name = 'capacity')
        
        flow = nx.min_cost_flow(GC)
        
        for node, value in flow.items():
            if value:
                for target, amount in value.items():
                    flows[(int(node), int(target))] = amount
        
        pd_flow = pd.DataFrame.from_dict(flows, columns = ['flow'], orient = 'index')
        pd_edges = pd_edges.merge(pd_flow, left_index = True, right_index = True, how = 'left').fillna(0)
        pd_edges['flow'] = pd_edges['flow'].fillna(0)        
        findMinimumModule(pd_edges, colToTake = 'flow', colToSave = 'z_prime')
    
    return pd_edges

def calcCostsDict(G):
    costsDict = dict()
    for u, v, attr in G.edges(data=True):
        edge = (u, v)
        costsDict[edge] = dict()
        for i in range(3):
            costsDict[edge][i] = attr['weight'] * i
    return costsDict

def computeMinimizationResult(sum_of_prizes, G, solution):
    result = sum_of_prizes
    res_nodes = set()
    for (u,v) in solution:
        result += G.get_edge_data(u, v)['weight']
        res_nodes.add(u)
        res_nodes.add(v)
    for u in G.nodes():
        if u in res_nodes:
            result -= G.nodes[u]['prize']
    
    return result
