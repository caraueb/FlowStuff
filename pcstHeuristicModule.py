import sys, os
sys.path.append(os.path.abspath('..'))
import pcst_flow_LP
import networkx as nx
from gurobipy import *
import random
from heuristicsHelper import *
from helpers import computeMinimizationResult as computeMinRes
import random
import sys, os
import pandas as pd
import random
from datetime import datetime
import heuristicConfig as conf
import time
import numpy as np
# %load_ext autoreload
# %autoreload 2

# Compute the LP Solution as a starting solution for the heuristic
def computeLP(G, Graph, terminals):
    model, var2edge = pcst_flow_LP.create_model(Graph, terminals)
    model.Params.TimeLimit = 200
    model.Params.Threads = 3
    model.optimize() 

    solution = pcst_flow_LP.extractSolution(Graph, model)

    solutionEdges = pcst_flow_LP.extractSolutionEdges(Graph, model)
    solutionNodes = pcst_flow_LP.extractSolutionNodes(Graph, model)
    prizes = sum([G.nodes[node].get('prize', 0) for node in G.nodes()])
    result = computeMinRes(prizes, G, solution)
    
    return solutionEdges, solutionNodes, result

def networkConstruction(G, terminals, g_star = None, y_var = None, solutionEdges = None, solutionNodes = None, root = None):
    
    demandNodes = []
    edge_amount = G.number_of_edges()
    node_amount = G.number_of_nodes()
    # Create Pandas DataFrame with EdgeList. Contains columns 'start','target','attributes'
    pd_edges = nx.to_pandas_edgelist(G)
    
    # Add the Edge Column by combining source and target
    pd_edges['edge'] = list(zip(pd_edges.source, pd_edges.target))
    
    # Make a Nodes DataFrame
    pd_nodes = pd.DataFrame(G.nodes(data=True), columns = ['node', 'prizes'])
    pd_nodes = pd.concat([pd_nodes.drop(['prizes'], axis=1), pd.json_normalize(pd_nodes['prizes'])], axis=1)
    
    # Change the indeces to edges and nodes
    pd_edges = pd_edges.set_index('edge')
    pd_nodes = pd_nodes.set_index('node')

    # If it's the first iteration, this part if is chosen.
    if g_star == None:
        # Initialize values that need to be filled in the former loops
        # T is not needed here but can be initialised already
        pd_edges = pd_edges.assign(requiredCap=np.zeros(edge_amount), z = np.zeros(edge_amount), g = np.zeros(edge_amount), solVal = np.zeros(edge_amount), T = np.zeros(edge_amount))

        # Set the solution value of the edges
        pd_solEdges = pd.DataFrame(solutionEdges, columns = ['edge', 'solVal', 'weight', 'capacity'])
        pd_solEdges = pd_solEdges.set_index('edge').fillna(0)
        pd_solEdges.drop(['weight','capacity'], axis = 1)
        pd.merge(pd_edges, pd_solEdges, left_index=True, right_index = True)
        pd_edges['requiredCap'] = pd_edges['solVal'] * pd_edges['capacity']


    else:
        # Extract the required Capacity from the g_star which has been given
        # to the function as an argument
        # Not entirely sure whether this is working
        pd_edges = pd_edges.assign(g = np.zeros(edge_amount), requiredCap = np.zeros(edge_amount))
        for k, v in g_star.items():
            pd_edges.at[k, 'requiredCap'] = v
    
    # Set the required capacity for the edge
    pd_edges = findMinimumModule(pd_edges)

    # Every selected customer has a demand of one that needs to be set. Since we round the solution from the LP,
    # we set the demand of the customer to their whole demand.
    # For every node included in the solution, i.e. Terminals with a value 1, we have to do something later.
    # Thus, add it to a set and set it's demand to a reasonable value, i.e. 1.
    pd_nodes = pd_nodes.assign(y=np.zeros(node_amount), b = np.zeros(node_amount))
    if y_var == None:
        pd_edges = pd_edges.assign(b=np.zeros(edge_amount), y = np.zeros(edge_amount))
        for node in solutionNodes:
            pd_nodes.loc[pd_nodes.index == node, ['y', 'b']] = 1
            demandNodes.append(node)
    else:
        pd_nodes['y'] = pd_nodes.index.map(y_var)
        pd_nodes.loc[pd_nodes['y'] > 0, 'b'] = 1
        demandNodes = pd_nodes[pd_nodes['y'] > 0.5].index.values.tolist()
     
    
    # While there is a node with a demand greater than 0
    while len(demandNodes) != 0:
        # Pick a random node with a demand
        node = random.choice(demandNodes)

        # Set the demand to be this node's demand
        demand = pd_nodes.loc[pd_nodes.index == node, 'b'].values[0]

        # Set that nodes demand to 0, and delete it from the Set of DemandNodes
        pd_nodes.loc[pd_nodes.index == node, 'b'] = 0
        demandNodes.remove(node)
        
        # Compute the weight Graph with the additional demand
        weightGraph = compEdgeWeights(G, pd_edges.copy(), demand, "z", "g")
        
        # Compute the shortest path for the chosen Node in the weight Graph
        spEdges = createShortestPath(weightGraph, root, node, 'cost')

        # Create DataFrame only with the edges that exist in the ShortestPath
        pd_shorPath = pd_edges[pd_edges.index.isin(spEdges)].copy()
        amountSpEdges = pd_shorPath.shape[0]

        conditions = [(pd_shorPath['requiredCap'] + demand > 0) & pd_shorPath['requiredCap'] + demand < conf.copperCap,
                pd_shorPath['requiredCap'] + demand > conf.copperCap]
        choices = [conf.copperCap - pd_shorPath['requiredCap'], conf.gfCap - pd_shorPath['requiredCap']]
        pd_shorPath = pd_shorPath.assign(remCap=np.zeros(amountSpEdges))
        
        pd_shorPath['remCap'] = np.select(conditions, choices)
        
        pd_shorPath['g'] += np.where(pd_shorPath['remCap'] >= demand, demand, pd_shorPath['remCap'])
        pd_shorPath = findMinimumModule(pd_shorPath, colToTake = 'g')
        
        pd_edges.update(pd_shorPath)
    
    
    pd_edges = calculateFlow(G, pd_edges, pd_nodes, terminals, root)
    return pd_edges, pd_nodes


def networkImprovement(G, pde, pdn, terminals, root):
    # Counter
    s = 0
    iteration = 0
    startTime = datetime.now()
    timeElapsed = 0
    prizes = sum([G.nodes[node].get('prize', 0) for node in G.nodes()])
    result = 0
    node_amount = G.number_of_nodes()
    edge_amount = G.number_of_edges()
    
    while s < 20:
        # Copying the dataFrames. This is important if no better solution could be found.
        pd_edges = pde.copy()
        pd_nodes = pdn.copy()
        flowPerCustomer = dict()
        maxEdge = (0,0)
        g_prime = dict()
        
        pd_nodes['y_prime'] = pd_nodes['y']
        pd_edges['f_prime'] = pd_edges['flow']
        
        pdflows = pd_edges[['source','target','flow']].copy()
        pdflows = pdflows[pdflows.flow > 0]
        y = pd_nodes[pd_nodes['y'] > 0].to_dict()['y']
        
        # Flow decomposition
        # Go through all flows
        # TODO MAKE FASTER

        # Filter out all edges which have no flow
        pdflows = pdflows[pdflows['flow'] > 0]
        
        pd_edges.assign(termFlow=np.zeros(edge_amount))
        pdterms = pd.DataFrame(columns = ['term', 'edge'])

        for term, value in y.items():
            nodeToSearch = term
            edges = []
            while(nodeToSearch != root):
                row = pdflows.index[(pdflows['target'] == nodeToSearch) & pdflows['flow'] > 0].tolist()
                nodeToSearch = pdflows.at[row[0], 'source']
                pdflows.at[row[0], 'flow'] -= 1
                edge = (pdflows.at[row[0], 'source'], pdflows.at[row[0], 'target'])
                
                # Add information about the edge and the terminal to a dataframe
                pdterms.loc[len(pdterms.index)] = [term, edge]

                # If the node we are searching for, i.e. the preceding node of the previous node, was found, add it to the path
                if edge not in flowPerCustomer:
                    flowPerCustomer[edge] = []
                flowPerCustomer[edge].append(term)
                # pd_edges[pd_edges.index == edge, 'termFlow'] += term
                if pdflows.at[row[0], 'flow'] == 0:
                    pdflows.drop(row[0], inplace = True)
        
        # Iterate over all edges that transport flow
        if iteration == 1 or iteration == 3:
            # Find the edge with the maximum weight. At the end of the Loop, the edge with the maximum weight is stored
            pd_edges['max_weight'] = pd_edges[((pd_edges['weight'] * pd_edges['z'])/pd_edges['flow']) * ((pd_edges['T'] + 1)%2)]
        else:
            # Find the edge with the maximum weight. At the end of the Loop, the edge with the maximum weight is stored
            pd_edges['max_weight'] = pd_edges['weight'] * pd_edges['z'] * ((pd_edges['T'] + 1)%2)
        
        maxEdge = pd_edges[pd_edges.max_weight == pd_edges.max_weight.max()].index[0]
        pde.at[pd_edges.max_weight == pd_edges.max_weight.max(), 'T'] = 1
        
        # Only fits for a certain problem instance. 
        # Stops the program if no further edge can be found that's not yet in the Tabu List.
        if maxEdge == (12,13):
            break
            
        # Do reduction now. Every edge consist of two nodes. Iterate through those nodes, find the customers whose flow
        # is transported via these Nodes. 
        # Customers on both nodes on the maxEdge
        customers = flowPerCustomer[maxEdge]

        # Iterate through all customers having flow along the maxEdge
        # TODO MAKE FASTER
        pd_nodes.loc[pd_nodes.index.isin(customers), 'y_prime'] = 0
        
        pdterms = pdterms[pdterms['term'].isin(customers)]
        edgesToRemove = pdterms['edge'].tolist()
        
        # An improvement could be to define a dataframe, e.g. pdterms, where each row contains a column for the terminal and another for the edge the terminal is on.
        # This dataframe contains for some terminal x as many rows as there are edges on the path from the root to x.
        # Filtering and finding the edges could be done much faster this way.
        # This improvement has already been done (legacy code is commented out), but a check has to be done whether this change is indeed correct.
        """ for customer in customers:
            # On all edges in the graph, reduce the flow of an edge if that customers' flow is transported via this edge
            for edge, terms in flowPerCustomer.items():
                if (customer in terms):
                    pd_edges.loc[pd_edges.index == edge, 'f_prime'] -= 1 """
        
        pd_edges.loc[pd_edges.index.isin(edgesToRemove), 'f_prime'] -= 1
        
        f_prime = pd_edges[pd_edges['f_prime'] > 0].to_dict()['f_prime']
        
        pd_edges = pd_edges.assign(g_prime = np.zeros(edge_amount))
        
        for edge, value in f_prime.items():
            g_prime[edge] = f_prime[edge]

            if (edge[1], edge[0]) in f_prime:
                g_prime[edge] += f_prime[(edge[1], edge[0])]

        # Find the weighted Graph with the demands of the customers.
        weightedGraph = compEdgeWeights(G, pd_edges, 1, 'z_prime', 'g_prime')

        pd_nodes = pd_nodes.assign(y_pprime = np.zeros(node_amount))
        y_pprime = pd_nodes['y_pprime'].to_dict()
        y_prime = pd_nodes['y_prime'].to_dict()
        
        # For all customers
        for node in terminals:
            if y_prime[node] == 0 and y_pprime[node] == 0:
                pd_nodes.loc[node,'spLength'] = nx.shortest_path_length(weightedGraph, root, node, 'weight')
                
                if iteration == 2 or iteration == 3:
                    pd_nodes['ktilde'] = pd_nodes['prize']/(pd_nodes['cost'] + pd_nodes['spLength'])
                else:
                    pd_nodes['ktilde'] = pd_nodes['spLength']

        
        # Since our desired price p0 is that of all terminals set every terminals value to 1
        while ((pd_nodes['prize'] * (pd_nodes['y_prime'] + pd_nodes['y_pprime'])).sum() < prizes * conf.alpha):
            chosen = pd_nodes[pd_nodes['ktilde'] == pd_nodes['ktilde'].max()].index[0]
            pd_nodes.loc[pd_nodes.index == chosen, ['y_pprime', 'ktilde']] = [1, 0]
        y_pprime = pd_nodes['y_pprime'].to_dict()    
        
        z = pd_edges[pd_edges['z'] > 0].to_dict()['z']
        y = pd_nodes[pd_nodes['y'] > 0].to_dict()['y']

        profitOld, chosenEdgesOld = calculateCostsFromFlow(z, y, G)
        t10 = time.time()
        pd_edgesNew, pd_nodesNew = networkConstruction(G, terminals, g_star = g_prime, y_var = y_pprime, root = root)
        print("construction again " + str(time.time() - t10))
        for node in terminals:
            y_prime[node] += y_pprime[node]

        z_prime = pd_edgesNew['z_prime'].to_dict()
        f_prime = pd_edgesNew['flow'].to_dict()

        profitNew, chosenEdges = calculateCostsFromFlow(z_prime, y_prime, G)
        result = computeMinimizationResult(prizes, G, chosenEdges)
        
        
        if profitOld < profitNew:
            pde['z'] = pd_edgesNew['z_prime']
            pdn['y'] = pd_nodes.index.map(y_prime)
            pde['flow'] = pd_edgesNew.index.map(f_prime)
            pde = pde.assign(T = np.zeros(edge_amount))
            s = 0
            result = computeMinimizationResult(prizes, G, chosenEdges)
            timeElapsed = datetime.now() - startTime
            with open('output1.txt', 'a') as f:
                f.write(str(result)+"\n")
        else:
            s += 1
            with open('output1.txt', 'a') as f:
                f.write(str(profitOld)+"\n")
            result = computeMinimizationResult(prizes, G, chosenEdgesOld)
        with open('output1.txt', 'a') as f:
                f.write(str(s)+"\n")

    
    return result, timeElapsed


