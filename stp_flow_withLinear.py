# +
from gurobipy import Model, GRB, quicksum
import networkx as nx

def create_model(G, terminals,weight='weight', minCapacity=0,  root = 1, pos = None):
    r""" Create an ILP for the Steiner tree problem with additional flow constraints

    Flow constraints limit the maximum amount of fLow from a specified root to the terminals.
    Each edge has a limit of flow it can transport.
    This flow can be extended to a larger amount up to each edges capacity.

    :param G: an ILPGraph
    :param terminals: a list of nodes that need to be connected by the Steiner tree
    :param weight: name of the argument in the edge dictionary of the graph used to store edge cost
    :param minCapacity: Current amount of Capacity that exists on each edge

    :return: a Gurobi model

    ILP:
        .. math::
            :nowrap:

            \begin{align*}
            \min \sum_{(u,v) \in \overrightarrow{E}} w_{uv} x_{uv}\\
            \text{s.t.} &&\\
            \forall (u,v) \in E: x_{uv} + x_{vu} \leq 1 \in E && \text{(at most one direction per edges)}\\
            \forall i \in V: x_{i}-\sum_{u=i \vee v=i}x_{uv} \leq 0 && \text{(prohibit isolated vertices)}\\
            \forall (u,v) \in E: n x_{(u,v)} + \ell_v - \ell_u \geq 1 - n (1-x_{vu}) && \text{(enforce increasing labels)}\\
            \forall (u,v) \in E: n x_{(v,u)} + \ell_u - \ell_v \geq 1 - n (1-x_{uv}) && \text{(enforce increasing labels)}\\
            \sum_{j \in O(r)} f_{i, j} = \sum_{i} b_i && \text{(flow starts at root}\\
            && \text{and ends at terminals)} \\
            \forall j \in V \backslash T: \sum_{i, j \in E} f_{i, j} - \sum_{j, k \in E} f_{j, k} = 0
            && \text{(flow conservation)}\\
            \forall j \in T: \sum_{i, j \in E} f_{i, j} - \sum_{j, k \in E} f_{j, k} = b_i
            && \text{(flow conservation)}\\
            \forall (i,j) \in E: f_{ij} \leq y_{ij} \kappa_{ij} + \kappa^0_{ij} &&
            \text{(there can only be flow when edge}\\
            && \text{is activated (limited by capacity))} \\
            \forall (i,j) \in E: 2(x_{ij}+x_{ji}) - x_i - x_j \leq 0 &&
            \text{(choose vertices when}\\
            && \text{edge is chosen)} \\
            \forall t \in T: x_{t} = 1 && \text{(all terminals must be chosen)}\\
            \end{align*}
    """
    # ensure that input is a directed graph
    if type(G.G) != nx.classes.digraph.DiGraph:
        G.G = nx.DiGraph(G.G)

    # create model
    m = Model("Steiner Tree")
    tot_load = 0
    for node in G.G.nodes(data=True):
        if (node[0] == root):
            continue
        if(node[0] in terminals):
            tot_load += 1

    n = G.G.number_of_nodes()

    # create reverse edge for every edge in the graph
    for edge in G.G.edges():
        reverseEdge = edge[::-1]
        if (reverseEdge not in G.G.edges()):
            G.G.add_edge(*edge[::-1])
            
    # Create Neighbourhood of the Root
    from_root = set()
    in_root = set()
    for edge in G.G.edges():
        if edge[0] == root:
            from_root.add(edge)
            in_root.add(edge[::-1])
        elif edge[1] == root:
            from_root.add(edge[::-1])
            in_root.add(edge)
    from_root = list(from_root)
    in_root = list(in_root)
    
    G.set_node_vars(m.addVars(G.G.nodes(), vtype=GRB.BINARY))
    G.set_edge_vars(m.addVars(G.G.edges(), vtype=GRB.BINARY))

    G.set_label_vars(m.addVars(G.G.nodes(), vtype=GRB.INTEGER, lb=1, ub=n))
    G.flow_variables = m.addVars(G.G.edges(), vtype=GRB.INTEGER, lb=0)

    m.update()

    # abbreviations
    edges = G.edge_variables
    nodes = G.node_variables
    labels = G.label_variables
    flow = G.flow_variables
    edge2var = dict(zip(edges.keys(), edges.values()))

    # set objective: minimise the sum of the weights of edges selected for the solution
    m.setObjective(quicksum([edge_var * 1 for edge, edge_var in edges.items()]), GRB.MINIMIZE)
    # G.G.edges[edge].get(weight, 1)
    
    # at most one direction per edge can be chosen
    # runtime can probably be greatly improved if iterating is done in a smarter way
    for edge, edge_var in edges.items():
        reverse_edge = edge[::-1]
        rev_edge_var = edge2var.get(reverse_edge)
        m.addConstr(edge_var + rev_edge_var <= 1)

    # prohibit isolated vertices
    for node, node_var in nodes.items():
        edge_vars = []
        for edge, edge_var in edges.items():
            # If the node is startpoint or endpoint of the edge, add the edge
            # to the array of edge variables
            if (node == edge[0] or node == edge[1]):
                edge_vars.append(edge_var)
        m.addConstr(node_var - quicksum(edge_vars) <= 0)

    # labeling constraints: enforce increasing labels in edge direction of selected edges
    for edge, edge_var in edges.items():
        reverseEdge = edge[::-1]
        edge_var_rev = edge2var.get(reverseEdge)
        m.addConstr(n * edge_var_rev + labels[edge[0]] - labels[edge[1]] >= 1 - n*(1 - edge_var))
        m.addConstr(n * edge_var + labels[edge[1]] - labels[edge[0]] >= 1 - n*(1 - edge_var_rev))

    # All flow has to start from the Root
    m.addConstr(gurobipy.quicksum(f[edge] for edge in Neighbourhood if edge[0] == root) == Terminals - 1)   
    
    # No flow to the root is allowed
    for edge in Neighbourhood:
        if edge[1] == root:
            f[edge] == 0
    
    # Flow conservation constraint
    for j in nodes:
        incomingEdges = G.G.in_edges(j)
        outgoingEdges = G.G.out_edges(j)
        
        # Everything that's not a root or terminal doesn't consume supply
        if j != root and j not in terminals:
            m.addConstr(sum(f[edgeIn] for edgeIn in incomingEdges) - sum(f[edgeOut] for edgeOut in outgoingEdges) == 0)
        
        # Flow conservation for Terminals. Terminals consume one unit of supply
        # This is a very simple assumption and subject to later change
        elif j != root:
            m.addConstr(sum(f[edgeIn] for edgeIn in incomingEdges) - sum(f[edgeOut] for edgeOut in outgoingEdges) == 1)

    return m

def extract_solution(G, model):
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
# -


