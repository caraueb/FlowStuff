import sys, os
sys.path.append(os.path.abspath('..'))
import pcst_flow_LP
import networkx as nx
from gurobipy import *
import random
from heuristicsHelper_loops import *
from helpers import computeMinimizationResult as cmr
import random
import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import random
from datetime import datetime
from heuristicConfig import *
import heuristicConfig as heuConf
# %load_ext autoreload
# %autoreload 2


def computeLP(G, Graph, terminals):
    model, var2edge = pcst_flow_LP.create_model(Graph, terminals)
    model.Params.TimeLimit = 20
    model.Params.Threads = 3
    model.optimize() 

    solution = pcst_flow_LP.extractSolution(Graph, model)

    solutionEdges = pcst_flow_LP.extractSolutionEdges(Graph, model)
    solutionNodes = pcst_flow_LP.extractSolutionNodes(Graph, model)
    prizes = sum([G.nodes[node].get('prize', 0) for node in G.nodes()])
    result = cmr(prizes, G, solution)
    
    return solutionEdges, solutionNodes, result

def networkConstruction(G, terminals, g_star = None, y_var = None, solutionEdges = None, solutionNodes = None, root = None):
    r"""
    :param G: A networkx Graph.
    :param terminals: The Terminals of the networkx graph
    :param g_star: Minimum required capacity for each edge in the graph.
    :param y_var: An array containing information about the y variable of every edge , i.e., whether the edge is chosen or not.
    :param solutionEdges: Edges chosen in the optimal solution of the LP Relaxation.
    :param solutionNodes: Nodes chosen in the optimal solution of the LP Relaxation.
    :param root: Root node of the Graph

    Network Construction Phase of the heuristic of Ivana Ljubic. 
    The Network Construction is on the one hand called to construct a feasible solution at the start and after improving the network.
    """
    
    requiredCaps = dict()
    demandNodes = []

    if y_var == None:
        y = dict()
    else:
        y = y_var
    
    g = dict()
    f = dict()
    z = dict()
    b = dict()
    #costsDict = calcCostsDict(G)

    # If it's the first iteration, this part if is chosen.
    if g_star == None:
        # Inititialization of all the values.
        for edge in G.edges():
            requiredCaps[edge] = 0
            z[edge] = 0
            g[edge] = 0
        
        # The solution of the LP relaxation contains all the information about the network.
        for edge in solutionEdges:
            edge = edge[0]
            # value is the continuous value of the edge variable. It's a value between 0 and 1, depending on how much of the capacity of an edge is needed. 
            value = edge[1]
            
            # Calculate an edge's required capacity by multiplying the edges capacity with its LP Relaxation total Flow. 
            reqCap = value * G.edges[edge]['capacity']
            
            # requiredCaps is a dictionary containing the required Capacity for each edge.
            requiredCaps[edge] = reqCap
            
            # Finding minimum module for required capacity on that edge.
            z[edge] = findMinimumModule(G, reqCap, edge)
    # ... if coming from the Network Improvement
    else:
        for edge in G.edges():
            # The required capacity for this edge is set to be the minimum required capacity g_star.
            requiredCaps[edge] = g_star[edge]
            # Finding the module for the minimum required capacity
            z[edge] = findMinimumModule(G, g_star[edge], edge)
            # g is set to be 0 as initialiation value for every Edge.
            g[edge] = 0
    
    # Every selected customer has a demand of one that needs to be set. Since we round the solution from the LP,
    # we set the demand of the customer to their whole demand in case the node is included in the solution, i.e., has a value > 0
    # In a first step all demand is set to zero, i.e., all y variables are set to 0.
    # In a second step, we iterate through all terminals which were chosen in the LP Relaxation solution and set their demand to 1.
    if y_var == None:
        for node in G.nodes:
            b[node] = 0
        for node in terminals:
            y[node] = 0
        for node in solutionNodes:
            b[node] = 1
            y[node] = 1
            demandNodes.append(node)
    # ... if coming from the Network Improvement.
    # Set the demand to 1 only if the has a y value greater than 0 and add the terminal to the demandNodes.
    else:
        for node, value in y.items():
            if value > 0:
                b[node] = 1
                demandNodes.append(node)

    # While there is a node with a demand greater than 0
    while len(demandNodes) != 0:
        # Pick a random node with a demand
        node = random.choice(demandNodes)
        
        # Set the demand to be this node's demand.
        # The variable "demand" helps us to figure out how the Graph and it's modules would look like if this nodes demand was added to each edge.
        # Depending on the cost increase with the transportation of demand of this node, we can figure out which node should be added to the solution.
        demand = b[node]
        
        # Set that nodes demand to 0, and delete it from the Set of DemandNodes
        b[node] = 0
        demandNodes.remove(node)

        # Compute the weight Graph with the additional demand artificially added to each edge.
        weightGraph = compEdgeWeights(G, g, demand, z)
        
        # Compute the shortest path for the chosen Node in the weight Graph, i.e., the one with the cheapest cost
        # spEdges then contains the edges of the shortest path
        spEdges = createShortestPath(weightGraph, root, node, 'weight')
        
        remCap = 0
        
        for edge in spEdges: 
            # If the minimum Capacity is exceeded, we have to install glass fibre. Thus the remaining capacity is 
            # increased in this case.
            # In case we install glass fibre, the remaining capacity is infinite.
            if requiredCaps[edge] + demand > minCapacity:
                remCap = heuConf.gfCap - requiredCaps[edge]
                
            # If no glass fibre is installed, remaining capacity
            else:
                remCap = heuConf.copperCap - requiredCaps[edge]
            # Get maximimum flow that can be transported on that edge.
            # The maximum is the minimum value of demand and remaining capacity of the edge. 
            maxTransport = min(demand, remCap)
            # Required capacity of the edge needs to be increased by the additional flow amount.
            g[edge] += maxTransport
            # Find the minimum module that can  transport that demand / flow
            z[edge] = findMinimumModule(G, g[edge], edge)
            
            #
            # Insufficient remaining capacity will never happen, thus this part of Ljubics algorithm has not been adde to the heuristic
            #
    # At the end of the network construction, the only thing that remains is calculating the flow on each edge.
    # The flow in the Graph depends on the terminals, the modules (z) on each edge and the chosen nodes (y).
    z, f = calculateFlow(G, z, terminals, y, root)

    return z, y, f


def networkImprovement(G, z, y, f, terminals, root):
    # Counter Tabu List
    s = 0
    # Taboo List
    T = []
    iteration = 0
    startTime = datetime.now()
    timeElapsed = 0
    # Sum of prizes for the recalculation of the objective function value. Needs to be done to make solutions comparable with the ones of Ivana Ljubic.
    prizes = sum([G.nodes[node].get('prize', 0) for node in G.nodes()])
    result = 0
    
    while s < 25:
        flowPerCustomer = dict()
        y_prime = dict()
        y_pprime = dict()
        z_prime = dict()
        g_prime = dict()
        maxWeight = 0
        maxEdge = (0,0)
        f_prime = dict()
        customerLengths = dict()
        ktilde = dict()

        # Initialization
        for edge in G.edges:
            z_prime[edge] = 0
            g_prime[edge] = 0

        for key, value in y.items():
            y_prime[key] = value

        for node in terminals:
            y_pprime[node] = 0

        # Initialization of f_prime.
        f_prime = f.copy()    
        flows = []

        # Flow decomposition.
        # Flows is a list of tuples. Every tuple contains the edge and the flow value of the edge in the following order
        # start, end, flowValue
        for edge, value in f.items():
            if value > 0:
                flows.append((edge[0], edge[1], value))
        # Pandas Dataframes of the Flows. This makes working with the flow decomposition easier.
        pdflows = pd.DataFrame(flows, columns = ['start', 'end', 'value'])

        # In this step we want to find the amount of flow of each customer on each edge.
        # In the end, each edge should be attributed with an amount of flow which flows
        # along that edge as well as how much flow of each customer flows along the edge.
        for term, value in y.items():
            if value > 0:
                # We are "following" the path of each terminal up to the root.
                # At first, we take some arbitrary terminal. Next we find the edge which transports flow to the terminal.
                # We find the originating node of that flow and search for the node which transports flow to the former source node and so on.
                # Step by step, we reach the root node. 
                # By doing this, each edge is attributed with how much from which customer flows along that edge.
                nodeToSearch = term
                while(nodeToSearch != root):
                    # Find the row where the searched Node is the node in the "end" column
                    row = pdflows.index[pdflows['end'] == nodeToSearch].tolist()
                    # In the next iteration, we are searching for the starting node of that edge
                    nodeToSearch = pdflows.at[row[0], 'start']
                    # Reduce flow of that edge by 1, i.e., the customer demand.
                    pdflows.at[row[0], 'value'] -= 1
                    # create edge tuple
                    edge = (pdflows.at[row[0], 'start'], pdflows.at[row[0], 'end'])

                    # If no flow was recorded for that edge yet, we have to create a new list entry for that edge
                    if edge not in flowPerCustomer:
                        flowPerCustomer[edge] = []
                    # Add the terminal to the nodes which transport flow along to edge
                    flowPerCustomer[edge].append(term)
                    # Remove the row if no flow is transported along that edge anymore, i.e., it's value is 0
                    if pdflows.at[row[0], 'value'] == 0:
                        pdflows.drop(row[0], inplace = True)

        # Finding the flow with the maxWeight
        # Depending on which iteration the algorithm is in, the weight is calculated differently.
        # This variation procedure widens the search range and thus can ensure better solutions.
        for edge, value in f.items():
            # First variant
            if iteration == 1 or iteration == 3:
                # Find the edge with the maximum weight. At the end of the Loop, the edge with the maximum weight is stored
                if ((G.edges[edge]['weight'] * z[edge])/value) > maxWeight and edge not in T:
                    maxWeight = (G.edges[edge]['weight'] * z[edge])/value
                    maxEdge = edge
            # Second variant
            else:
                # Find the edge with the maximum weight. At the end of the Loop, the edge with the maximum weight is stored
                if G.edges[edge]['weight'] * z[edge] > maxWeight and edge not in T:
                    maxWeight = G.edges[edge]['weight'] * z[edge]
                    maxEdge = edge

        # No edge could be found, since the default value (0,0) wasn't overwritten. End the Tabu search
        if maxEdge == (0,0):
            break
        
        # Reduction of the Graph.
        # Every edge consist of two nodes. Iterate through those nodes, find the customers whose flow is transported via these Nodes. 
        # customers are all the customers which have flow on the edge with the maximum flow.
        # Those customers are deleted from the graph in the next steps.
        customers = flowPerCustomer[maxEdge]
        
        # Iterate through all customers having flow along the maxEdge
        for customer in customers:
            # Set the decision variable of this customer to 0, i.e. deselect this customer
            y_prime[customer] = 0

            # On all edges in the graph, reduce the flow of an edge if that customers' flow is transported flow via this edge
            for edge, terms in flowPerCustomer.items():
                if (customer in terms):
                    # Remove flow caused by that customer on all edges of f_prime
                    try:
                        f_prime[edge] -= 1
                        # If now flow is transported via this edge anymore, delete the edge from the graph.
                        if f_prime[edge] == 0:
                            del f_prime[edge]
                    # If there is no flow of that customer on that edge, go to next edge
                    except:
                        pass
        
        
        # Calculate required Capacity after having removed the customers from maxEdge
        # Initialization
        for edge in G.edges():
            g_prime[edge] = 0
        # Calculate the required capacity for the edge.
        for edge, value in f_prime.items():
            g_prime[edge] = f_prime[edge]

            if (edge[1], edge[0]) in f_prime:
                g_prime[edge] += f_prime[(edge[1], edge[0])]

        # Find the weighted Graph with the demands of the customers. This requires uniform demand.
        weightedGraph = compEdgeWeights(G, g_prime, 1, z_prime)

        # Go through all terminals and find the terminals which are not yet included in the solution.
        # Calculate their shortest path length to the root
        for node in terminals:
            if y_prime[node] == 0 and y_pprime[node] == 0:
                spLength = nx.shortest_path_length(weightedGraph, root, node, 'weight')
                # Depending on the iteration the program is in, we either add the length of the shortest path
                # by calculating a helper value ktilde
                if iteration == 2 or iteration == 3:
                    ktilde[node] = (G.nodes[node]['prize']/(G.nodes[node].get('cost', 0) + spLength))
                # ... or by simply adding the length to the customerLengths array
                else:
                    customerLengths[node] = spLength
        
        # Adding some customers to the new solution by setting their y_pprime values to 1.
        # Keep adding customers until the reached prize is higher than some threshold value.
        # The threshold value is the minimum prize we want to achieve. 
        while sum(G.nodes[node]['prize']*(y_prime[node] + y_pprime[node]) for node in terminals) < prizes * alpha:
            if iteration == 2 or iteration == 3:
                chosen = max(ktilde, key = ktilde.get)
                y_pprime[chosen] = 1
                ktilde.pop(chosen)
            else:
                chosen = min(customerLengths, key=customerLengths.get)
                y_pprime[chosen] = 1
                customerLengths.pop(chosen)

        # Calculating the profit with the initial, old solution
        profitOld, chosenEdgesOld = calculateCostsFromFlow(z, y, G)
        
        # Constructing a network with the new solution
        z_prime, waste, f_prime = networkConstruction(G, terminals, g_star = g_prime, y_var = y_pprime, root = root)

        # Include all y_pprime customers, i.e., the ones which were added, to the new solution
        for node in terminals:
            y_prime[node] += y_pprime[node]

        # Calculate the new profit
        profitNew, chosenEdges = calculateCostsFromFlow(z_prime, y_prime, G)
        
        # If the new Profit is higher, reset the Tabu List and start with a new incumbent solution
        if profitOld < profitNew:
            z = z_prime
            y = y_prime
            f = f_prime
            T = []
            s = 0
            result = cmr(prizes, G, chosenEdges)
            timeElapsed = datetime.now() - startTime
            print(profitNew)
        # If not, start another iteration and add the maxEdge to the Tabu List
        else:
            s += 1
            T.append(maxEdge)
            
            print(profitOld)
        print(s)
        
    
    return result, timeElapsed


