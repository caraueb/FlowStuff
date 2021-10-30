from gurobipy import Model, GRB, quicksum
import networkx as nx


def create_model(G, terminals, weight='weight', warmstart=[], lower_bound=None):
    r""" Create an ILP for the minimum Steiner tree problem in graphs.

    """
    # create model
    m = Model("Steiner Tree")

    G.G = nx.DiGraph(G.G)

    n = G.G.number_of_nodes()
    T = len(terminals)
    root = terminals[0]

    # add variables for edges and nodes
    edge_flow = m.addVars(G.G.edges(), vtype=GRB.CONTINUOUS, lb=-T, ub=T)
    G.set_edge_vars(m.addVars(G.G.edges(), vtype=GRB.BINARY))

    m.update()

    # abbreviations
    edges = G.edge_variables

    # set objective: minimise the sum of the weights of edges selected for the solution
    m.setObjective(quicksum([edge_var * G.G.edges[edge][weight] for edge, edge_var in edges.items()]), GRB.MINIMIZE)

    # flow != 0 => edge chosen
    for edge, edge_var in edges.items():
        m.addConstr(2 * T * edge_var + edge_flow[edge] >= 0)
        m.addConstr(2 * T * edge_var - edge_flow[edge] >= 0)

    # flow condition on non-terminal nodes
    for node in G.G.nodes():
        if node not in terminals:
            m.addConstr(quicksum(edge_flow[e] for e in G.G.edges(node)) - quicksum(edge_flow[e] for e in G.G.in_edges(node)) == 0)

    # flow condition on root
    m.addConstr(quicksum(edge_flow[e] for e in G.G.edges(root)) - quicksum(edge_flow[e] for e in G.G.in_edges(root)) == -T+1)

    # flow conditions on non-root terminal
    for t in terminals:
        if t != root:
            m.addConstr(quicksum(edge_flow[e] for e in G.G.edges(t)) - quicksum(edge_flow[e] for e in G.G.in_edges(t)) == 1)

    m.update()

    return m


def extract_solution(G, model):
    r""" Get the optimal Steiner tree in G

        :param G: a weighted :py:class:`~graphilp.imports.ilpgraph.ILPGraph`
        :param model: a solved Gurobi model for the minimum Steiner tree problem

        :return: the edges of an optimal Steiner tree connecting all terminals in G
    """
    solution = []

    for edge, edge_var in G.edge_variables.items():
        if edge_var.X > 0.5:
            solution.append(edge)

    return solution
