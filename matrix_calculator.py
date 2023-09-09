import numpy as np
from sympy import symbols, Matrix

# Matrix definition
def create_matrix(rows, cols, symbolic=False, default_value=0):
    """
    Create a matrix of given dimensions with either numeric or symbolic entries.
    
    Parameters:
    - rows (int): Number of rows for the matrix.
    - cols (int): Number of columns for the matrix.
    - symbolic (bool): If True, the matrix will be symbolic; otherwise, numeric.
    - default_value (numeric): Default value for matrix entries if matrix is numeric.
    
    Returns:
    - Matrix: A symbolic matrix if symbolic=True; otherwise, a numeric matrix.
    """
    if symbolic:
        return Matrix(rows, cols, lambda i, j: symbols(f'a_{i+1}{j+1}'))
    else:
        return np.full((rows, cols), default_value)
    
# Check shape
def is_square(matrix):
    """
    Check if a matrix is square.
    
    Parameters:
    - matrix (Matrix or np.ndarray): The matrix to check.
    
    Returns:
    - bool: True if the matrix is square, False otherwise.
    """
    return matrix.shape[0] == matrix.shape[1]

def matrix_shape(matrix):
    """
    Retrieve the shape (dimensions) of a matrix.
    
    Parameters:
    - matrix (Matrix or np.ndarray): The matrix to check.
    
    Returns:
    - tuple: (number of rows, number of columns)
    """
    return matrix.shape

# Error Handling
def addition_compatibility(A, B):
    """
    Ensure two matrices are conformable for addition.
    
    Parameters:
    - A, B (Matrix or np.ndarray): Matrices to check.
    
    Raises:
    - ValueError: If matrices have different dimensions.
    
    Returns:
    - bool: True if matrices can be added, otherwise an exception is raised.
    """
    if matrix_shape(A) != matrix_shape(B):
        raise ValueError('Matrices are not of the same size and cannot be added')
    return True

def multiplication_compatibility(A, B):
    """
    Ensure two matrices are conformable for multiplication.
    
    Parameters:
    - A, B (Matrix or np.ndarray): Matrices to check.
    
    Raises:
    - ValueError: If number of columns in A doesn't match number of rows in B.
    
    Returns:
    - bool: True if matrices can be multiplied, otherwise an exception is raised.
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError('Number of columns in the first matrix must equal the number of rows in the second matrix for multiplication')
    return True

def check_square(matrix):
    """
    Ensure a matrix is square.
    
    Parameters:
    - matrix (Matrix or np.ndarray): Matrix to check.
    
    Raises:
    - ValueError: If the matrix is not square.
    
    Returns:
    - bool: True if the matrix is square, otherwise an exception is raised.
    """
    if not is_square(matrix):
        raise ValueError('The matrix is not square')
    return True

# Print symbolic output
def print_matrix(matrix):
    """
    Print a matrix in a formatted style.
    
    Parameters:
    - matrix (Matrix or np.ndarray): Matrix to print.
    
    Returns:
    - None
    """
    if isinstance(matrix, Matrix):
        for i in range(matrix.rows):
            row_str = "["
            for j in range(matrix.cols):
                row_str += f" {matrix[i, j]}"
            row_str += "]"
            print(row_str)
    else:
        print(matrix)

# # Example usage
# A = create_matrix(2, 3)
# B = create_matrix(3, 2)
# C = create_matrix(2, 3, symbolic=True)

# print_matrix(A)
# print_matrix(B)
# print_matrix(C)