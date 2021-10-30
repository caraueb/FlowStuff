# Recomputes the objective value of a PCST solution. This is done to make the objective function value comparable with the ones from Ivana Ljubic in her paper 
# An algorithmic framework for the exact solution of the prize-collecting Steiner tree problem (2006)
# She doesn't consider the total profit as the objective function but the minimal opporutunity costs. This method recomputes the solution.
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
    print(result)
    return result

# Recomputes the objective value in the same manner as computeMinimzationResults. The input data are, however, pcst_fast solution edges.
def computeMinimizationResultPCSTFast(sum_of_prizes, G, solutionEdges, edges):
    solutionEdges = solutionEdges.tolist()
    edges = list(edges)
    result = sum_of_prizes
    res_nodes = set()
    result = sum_of_prizes
    for edge in solutionEdges:
        actualEdge = edges[edge]
        result += G.get_edge_data(actualEdge[0], actualEdge[1])['weight']
        res_nodes.add(actualEdge[0])
        res_nodes.add(actualEdge[1])
        
    for u in G.nodes():
        if u in res_nodes:
            result -= G.nodes[u]['prize']
    print(result)
    return result