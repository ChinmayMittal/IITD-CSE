import numpy as np

def find_min_zero(zero_matrix, marked_zeros):
    min_zero = [np.inf, -np.inf]
    for row_idx in range(zero_matrix.shape[0]): 
        zero_count = np.sum(zero_matrix[row_idx] == True)
        if zero_count > 0 and min_zero[0] > zero_count:
            min_zero = [zero_count, row_idx]
    zero_col_idx = np.where(zero_matrix[min_zero[1]] == True)[0][0]
    marked_zeros.append((min_zero[1], zero_col_idx))
    zero_matrix[min_zero[1], :] = False
    zero_matrix[:, zero_col_idx] = False

def process_matrix(matrix):
    current_matrix = matrix
    zero_matrix = (current_matrix == 0)
    zero_matrix_copy = zero_matrix.copy()
    marked_zeros = []
    while True in zero_matrix_copy:
        find_min_zero(zero_matrix_copy, marked_zeros)
    marked_zero_rows = []
    marked_zero_cols = []
    for mark in marked_zeros:
        marked_zero_rows.append(mark[0])
        marked_zero_cols.append(mark[1])
    non_marked_rows = list(set(range(current_matrix.shape[0])) - set(marked_zero_rows))
    marked_columns = []
    change_flag = True
    while change_flag:
        change_flag = False
        for row in non_marked_rows:
            for col, is_zero in enumerate(zero_matrix[row]):
                if is_zero and col not in marked_columns:
                    marked_columns.append(col)
                    change_flag = True
        for row, col in marked_zeros:
            if row not in non_marked_rows and col in marked_columns:
                non_marked_rows.append(row)
                change_flag = True
    uncovered_rows = list(set(range(matrix.shape[0])) - set(non_marked_rows))
    return marked_zeros, uncovered_rows, marked_columns

def updateMatrix(matrix, covered_rows, covered_cols):
    uncovered_values = [matrix[row, col] for row in range(matrix.shape[0]) if row not in covered_rows for col in range(matrix.shape[1]) if col not in covered_cols]
    min_value = min(uncovered_values)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if row not in covered_rows and col not in covered_cols:
                matrix[row, col] -= min_value
            elif row in covered_rows and col in covered_cols:
                matrix[row, col] += min_value
    return matrix

def hungarianAlgorithm(input_matrix, find_min=True):
    if find_min:
        input_matrix = np.max(input_matrix) - input_matrix
    working_matrix = input_matrix.copy()
    for row in range(input_matrix.shape[0]): 
        working_matrix[row] -= np.min(working_matrix[row])
    for col in range(input_matrix.shape[1]): 
        working_matrix[:, col] -= np.min(working_matrix[:, col])
    zero_count = 0
    assignment_positions = None
    while zero_count < input_matrix.shape[0]:
        assignment_positions, covered_rows, covered_columns = process_matrix(working_matrix)
        zero_count = len(covered_rows) + len(covered_columns)
        if zero_count < input_matrix.shape[0]:
            working_matrix = updateMatrix(working_matrix, covered_rows, covered_columns)
    return [position[1] for position in assignment_positions]