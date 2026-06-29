"""
0/1 Knapsack Problem Solver using Dynamic Programming

This module provides a function to solve the 0/1 knapsack problem,
which is commonly used in video summarization to select the subset of
video shots/frames that maximizes the total importance score while remaining
within a maximum duration constraint (knapsack capacity).

Time complexity: O(n * W), where n is the number of items and W is the capacity.
"""

import numpy as np


def knapsack_dp(values, weights, n_items, capacity, return_all=False):
    """
    Solves the 0/1 knapsack problem using Dynamic Programming.

    Args:
        values (list of float/int): The values/scores of the items.
        weights (list of int): The weights (e.g. durations or frame counts) of the items.
        n_items (int): The total number of items available.
        capacity (int): The maximum weight capacity of the knapsack.
        return_all (bool, optional): If True, returns both the indices of the selected 
                                     items and the maximum value achieved. Defaults to False.

    Returns:
        list: 0-indexed list of integers indicating the indices of the selected items.
        tuple: (picks, max_val) if return_all is True, where picks is the list of indices
               and max_val is the maximum achieved value.
    """
    # Validate the inputs to prevent logical errors during DP execution
    check_inputs(values, weights, n_items, capacity)

    # Initialize the DP table of shape (n_items + 1, capacity + 1)
    # table[i, w] stores the maximum value that can be attained with a weight limit of w using the first i items.
    table = np.zeros((n_items + 1, capacity + 1), dtype=np.float32)
    
    # keep[i, w] is set to 1 if item i is selected in the optimal solution for weight limit w, and 0 otherwise.
    keep = np.zeros((n_items + 1, capacity + 1), dtype=np.float32)

    # Fill the DP table iteratively
    for i in range(1, n_items + 1):
        for w in range(0, capacity + 1):
            wi = weights[i - 1]  # Weight of the current item
            vi = values[i - 1]   # Value/score of the current item
            
            # If the item fits and choosing it yields a higher total value than not choosing it:
            if (wi <= w) and (vi + table[i - 1, w - wi] > table[i - 1, w]):
                table[i, w] = vi + table[i - 1, w - wi]
                keep[i, w] = 1
            else:
                table[i, w] = table[i - 1, w]

    # Backtrack through the 'keep' table to identify which items were selected
    picks = []
    K = capacity

    for i in range(n_items, 0, -1):
        if keep[i, K] == 1:
            picks.append(i)
            K -= weights[i - 1]  # Deduct the item's weight from the remaining capacity

    # Sort selected item indices
    picks.sort()
    # Convert from 1-based indexing used in the DP loop to standard 0-based indexing
    picks = [x - 1 for x in picks]

    if return_all:
        max_val = table[n_items, capacity]
        return picks, max_val
    return picks


def check_inputs(values, weights, n_items, capacity):
    """
    Validates input argument types and constraints to ensure correct DP execution.
    """
    # Check variable types
    assert isinstance(values, list), "values must be a list"
    assert isinstance(weights, list), "weights must be a list"
    assert isinstance(n_items, int), "n_items must be an integer"
    assert isinstance(capacity, int), "capacity must be an integer"
    
    # Check item value types
    assert all(isinstance(val, int) or isinstance(val, float) for val in values), "Each value must be an int or float"
    assert all(isinstance(val, int) for val in weights), "Each weight must be an integer"
    
    # Validate mathematical constraints
    assert all(val >= 0 for val in weights), "Weights must be non-negative"
    assert n_items > 0, "Number of items must be greater than 0"
    assert capacity > 0, "Capacity must be greater than 0"


if __name__ == '__main__':
    # Define a simple test case
    # Item 1: value 2, weight 1
    # Item 2: value 3, weight 2
    # Item 3: value 4, weight 3
    # Capacity: 3
    # Optimal choice: Item 1 + Item 2 (total weight 3, total value 5)
    values = [2, 3, 4]
    weights = [1, 2, 3]
    n_items = 3
    capacity = 3
    picks = knapsack_dp(values, weights, n_items, capacity)
    print("Selected item indices (0-indexed):", picks)
