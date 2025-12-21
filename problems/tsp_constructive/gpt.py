def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node by balancing the immediate cost with the average cost
    of moving from that candidate to all other unvisited nodes. This strategy
    favors nodes that are centrally located among the remaining options,
    potentially creating a more balanced path.
    """
    next_node = None
    best_score = float('inf')

    if not unvisited_nodes:
        return None

    # If only one unvisited node is left, it's the only choice.
    if len(unvisited_nodes) == 1:
        return list(unvisited_nodes)[0]

    # Iterate through each potential next node to evaluate it.
    for candidate_node in unvisited_nodes:
        # Cost of the first step (current -> candidate).
        cost_step1 = distance_matrix[current_node][candidate_node]
        
        # Find the average cost of the second step (candidate -> all other unvisited).
        nodes_for_step2 = unvisited_nodes - {candidate_node}
        
        # Calculate the average distance from the candidate to the remaining unvisited nodes.
        sum_of_dists_step2 = 0
        for next_hop_node in nodes_for_step2:
            sum_of_dists_step2 += distance_matrix[candidate_node][next_hop_node]
        
        avg_cost_step2 = sum_of_dists_step2 / len(nodes_for_step2) if nodes_for_step2 else 0

        # The score is the sum of the immediate cost and the average future cost.
        total_score = cost_step1 + avg_cost_step2
        
        if total_score < best_score:
            best_score = total_score
            next_node = candidate_node
            
    return next_node
