import numpy as np
from sympy import symbols, Matrix
from scipy.linalg import expm
from scipy.linalg import logm, expm
from numpy.linalg import matrix_rank
from numpy.linalg import cond
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

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

# Properties and error handling
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
    from matrix_calculator import is_square
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

# Addition
def matrix_add(A, B):
    """
    Add two matrices.
    
    Parameters:
    - A, B (Matrix or np.ndarray): Matrices to be added.
    
    Returns:
    - Matrix or np.ndarray: Resultant matrix after addition.
    """
    from matrix_calculator import addition_compatibility
    addition_compatibility(A, B)
    if isinstance(A, Matrix):
        return A + B
    else:
        return np.add(A, B)
     
# Subtraction
def matrix_subract(A, B):
    """
    Subtract matrix B from matrix A.
    
    Parameters:
    - A, B (Matrix or np.ndarray): Matrices involved in subtraction.
    
    Returns:
    - Matrix or np.ndarray: Resultant matrix after subtraction.
    """
    from matrix_calculator import addition_compatibility
    addition_compatibility(A, B)
    if isinstance(A, Matrix):
        return A - B
    else:
        return np.subtract(A, B)
    
# Multiplication
def matrix_multiply(A, B):
    """
    Multiply two matrices.
    
    Parameters:
    - A, B (Matrix or np.ndarray): Matrices to be multiplied.
    
    Returns:
    - Matrix or np.ndarray: Resultant matrix after multiplication.
    """
    from matrix_calculator import multiplication_compatibility
    multiplication_compatibility(A, B)
    if isinstance(A, Matrix):
        return A * B
    else:
        return np.dot(A, B)
    
# Transposition
def matrix_transpose(matrix):
    """
    Transpose a matrix.
    
    Parameters:
    - matrix (Matrix or np.ndarray): Matrix to be transposed.
    
    Returns:
    - Matrix or np.ndarray: Transposed matrix.
    """
    if isinstance(matrix, Matrix):
        return matrix.transpose()
    else:
        return np.transpose(matrix)
    
# Determinant
def matrix_determinant(matrix):
    """
    Calculate the determinant of a matrix.
    
    Parameters:
    - matrix (Matrix or np.ndarray): Matrix for which determinant is to be calculated.
    
    Returns:
    - number (float or Symbol): Determinant of the matrix.
    """
    from matrix_calculator import check_square
    check_square(matrix)
    if isinstance(matrix, Matrix):
        return matrix.det()
    else:
        return np.linalg.det(matrix)
    
# Inversion
def matrix_inverse(matrix):
    """
    Calculate the inverse of a matrix.
    
    Parameters:
    - matrix (Matrix or np.ndarray): Matrix to be inverted.
    
    Returns:
    - Matrix or np.ndarray: Inverse of the matrix.
    """
    from matrix_calculator import check_square
    check_square(matrix)
    if isinstance(matrix, Matrix):
        if matrix.det() == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
        return matrix.inv()
    else:
        if np.linalg.det(matrix) == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
        return np.linalg.inv(matrix)
    
# Eigenvalues and eigenvectors
def matrix_eigen(matrix):
    """
    Compute the eigenvalues and eigenvectors of a matrix.
    
    Parameters:
    - matrix (Matrix or np.ndarray): Matrix for which eigenvalues and eigenvectors are to be computed.
    
    Returns:
    - tuple: Eigenvalues and eigenvectors.
    """
    from matrix_calculator import check_square
    check_square(matrix)
    if isinstance(matrix, Matrix):
        return matrix.eigenvals(), matrix.eigenvects()
    else:
        return np.linalg.eig(matrix)
    
# LU decomposition
def matrix_lu_decomposition(matrix):
    """
    Perform LU decomposition on a matrix.
    
    Parameters:
    - matrix (Matrix or np.ndarray): Matrix to be decomposed.
    
    Returns:
    - tuple: Lower triangular matrix (L) and upper triangular matrix (U).
    """
    if isinstance(matrix, Matrix):
        L, U, _ = matrix.LUdecomposition()
        return L, U
    else:
        import scipy.linalg
        P, L, U = scipy.linalg.lu(matrix)
        return L, U
    
# Cholesky decomposition
def matrix_cholesky_decomposition(matrix):
    """
    Perform Cholesky decomposition on a matrix.
    
    Parameters:
    - matrix (Matrix or np.ndarray): Matrix to be decomposed.
    
    Returns:
    - np.ndarray or Matrix: Lower triangular matrix.
    """
    from matrix_calculator import check_square
    check_square(matrix)
    if isinstance(matrix, Matrix):
        return matrix.cholesky()
    else:
        return np.linalg.cholesky(matrix)
    
# QR decomposition
def matrix_qr_decomposition(matrix):
    """
    Perform QR decomposition on a matrix.
    
    Parameters:
    - matrix (Matrix or np.ndarray): Matrix to be decomposed.
    
    Returns:
    - tuple: Orthogonal matrix (Q) and upper triangular matrix (R).
    """
    if isinstance(matrix, Matrix):
        Q, R = matrix.QRdecomposition()
        return Q, R
    else: 
        return np.linalg.qr(matrix)
    
# Singular Value Decomposition (SVD)
def matrix_svd(matrix):
    """
    Perform Singular Value Decomposition on a matrix.
    
    Parameters:
    - matrix (Matrix or np.ndarray): Matrix to be decomposed.
    
    Returns:
    - tuple: U, Sigma (Σ), and V* matrices.
    """
    if isinstance(matrix, Matrix):
        U, S, V = matrix.singular_value_decomposition()
        return U, S, V
    else:
        return np.linalg.svd(matrix)

# Determinant
def matrix_determinant(matrix):
    """
    Compute the determinant of a given matrix.
    
    Parameters:
    - matrix (numpy.ndarray or sympy.Matrix): The input matrix.
    
    Returns:
    - float or sympy.Expr: The determinant of the matrix.
    """
    if isinstance(matrix, np.ndarray):
        return np.linalg.det(matrix)
    else:
        return matrix.det()
    
# Trace
def matrix_trace(matrix):
    """
    Compute the trace of a given matrix.
    
    Parameters:
    - matrix (numpy.ndarray or sympy.Matrix): The input matrix.
    
    Returns:
    - float or sympy.Expr: The trace of the matrix.
    """
    if isinstance(matrix, np.ndarray):
        return np.trace(matrix)
    else:
        return matrix.trace()
    
# Inverse
def matrix_inverse(matrix):
    """
    Compute the inverse of a given matrix if it exists.
    
    Parameters:
    - matrix (numpy.ndarray or sympy.Matrix): The input matrix.
    
    Returns:
    - numpy.ndarray or sympy.Matrix: The inverse of the matrix.
    """
    if isinstance(matrix, np.ndarray):
        return np.linalg.inv(matrix)
    else:
        return matrix.inv()
    
# Pseudoinverse
def matrix_pseudoinverse(matrix):
    """
    Compute the Moore-Penrose pseudoinverse of a given matrix.
    
    Parameters:
    - matrix (numpy.ndarray): The input matrix.
    
    Returns:
    - numpy.ndarray: The pseudoinverse of the matrix.
    """
    return np.linalg.pinv(matrix)

# Exponential
def matrix_exponential(matrix):
    """
    Compute the matrix exponential of a given matrix.
    
    Parameters:
    - matrix (numpy.ndarray): The input matrix.
    
    Returns:
    - numpy.ndarray: The matrix exponential.
    """
    return expm(matrix)

# Rank
def compute_rank(matrix):
    """Compute the rank of a matrix.
    
    Args:
    - matrix (Matrix): Input matrix (either symbolic or numerical).
    
    Returns:
    - int: Rank of the matrix.
    """
    return matrix_rank(matrix)

# Condition number
def compute_condition_number(matrix):
    """Compute the condition number of a matrix.
    
    Args:
    - matrix (Matrix): Input matrix (either symbolic or numerical).
    
    Returns:
    - float: Condition number of the matrix.
    """
    return cond(matrix)

# Norms
def compute_norm(matrix, norm_type="fro"):
    """
    Compute the norm of a matrix.
    
    Args:
    - matrix (ndarray): The input matrix.
    - norm_type (str): The type of norm to compute. Supported values are 'fro', 'l1', and 'linf'.
    
    Returns:
    - float: The computed matrix norm.
    """
    if norm_type == "fro":
        return np.linalg.norm(matrix,"fro")
    elif norm_type == "l1":
        return np.linalg.norm(matrix, 1)
    elif norm_type == "linf":
        return np.linalg.norm(matrix, np.inf)
    else:
        raise ValueError("Unsupported norm type. Use 'fro', 'l1' or 'linf'.")
    
# Kronecker product
def kronecker_product(matrix1, matrix2):
    """
    Compute the Kronecker product of two matrices.
    
    Args:
    - matrix1 (ndarray): The first input matrix.
    - matrix2 (ndarray): The second input matrix.
    
    Returns:
    - ndarray: The computed Kronecker product.
    """
    return np.kron(matrix1, matrix2)

# Matrix power
def matrix_power(matrix, exponent):
    """
    Raise a matrix to a given power.
    
    Args:
    - matrix (ndarray or Matrix): The input matrix.
    - exponent (float): The exponent to which the matrix is to be raised.
    
    Returns:
    - ndarray or Matrix: The matrix raised to the given power.
    
    Raises:
    - ValueError: If the matrix is not square.
    """
    if not is_square(matrix):
        raise ValueError("The matrix must be square for matrix powers.")
    
    if int(exponent) == exponent:
        return np.linalg.matrix_power(matrix, int(exponent))
    else:
        return expm(exponent * logm(matrix))    # for non-int powers, use A^r = exp(r*log(A))

# Matrix logarithm    
def matrix_logarithm(matrix):
    """
    Compute the logarithm of a matrix.
    
    Args:
    - matrix (ndarray or Matrix): The input matrix.
    
    Returns:
    - ndarray or Matrix: The matrix logarithm.
    
    Raises:
    - ValueError: If the matrix is not square or if it's singular.
    """
    if not is_square(matrix):
        raise ValueError("The matrix must be square for matrix logarithm.")
    if np.linalg.det(matrix) == 0:
        raise ValueError("Logarithm is undefined for singular matrices.")
    
    return logm(matrix)

# Linear equations
def solve_linear_system(matrix, b):
    """
    Solve a system of linear equations Ax = b.
    
    Args:
    - matrix (ndarray or Matrix): Coefficient matrix A.
    - b (ndarray or list): Right-hand side vector.
    
    Returns:
    - ndarray: Solution vector x.
    
    Raises:
    - ValueError: If the matrix is singular.
    """
    if np.linalg.det(matrix) == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    
    return np.linalg.inv(matrix).dot(b)

# Differential equations
def solve_matrix_differential(A, x0, t):
    """
    Solve a system of linear differential equations using matrices.
    
    Args:
    - A (ndarray): Coefficient matrix A.
    - x0 (ndarray or list): Initial conditions vector.
    - t (float): Time at which the solution is desired.
    
    Returns:
    - ndarray: Solution vector x(t).
    """
    return expm(A * t).dot(x0)

# Create Sparse Matrices
def create_sparse(matrix):
    """
    Convert a dense matrix to a compressed sparse row (CSR) matrix.

    Args:
    - matrix (2D list/2D numpy array): Input dense matrix.

    Returns:
    - csr_matrix: Compressed Sparse Row representation of the matrix.
    """
    return csr_matrix(matrix)

# Convert Sparse to Dense
def sparse_to_dense(sparse_matrix):
    """
    Convert a sparse matrix to its dense representation.

    Args:
    - sparse_matrix (csr_matrix): Input sparse matrix.

    Returns:
    - numpy.ndarray: Dense representation of the matrix.
    """
    return sparse_matrix.toarray()

# Check if sparse
def check_sparsity(matrix):
    """
    Check the sparsity of a matrix.

    Args:
    - matrix (2D list/2D numpy array/csr_matrix): Input matrix.

    Returns:
    - float: Percentage of non-zero entries in the matrix.
    """
    if not isinstance(matrix, csr_matrix):
        matrix = csr_matrix(matrix)
    total_elements = matrix.shape[0] * matrix.shape[1]
    non_zero_elements = matrix.nnz
    sparsity = (total_elements - non_zero_elements) / total_elements * 100
    return sparsity

# Get non-zero elements and their locations
def get_non_zero(sparse_matrix):
    """
    Get the non-zero elements and their row, column locations from a sparse matrix.

    Args:
    - sparse_matrix (csr_matrix): Input sparse matrix.

    Returns:
    - tuple: (data, (row_indices, col_indices))
    """
    return sparse_matrix.data, sparse_matrix.nonzero()

# Finite Difference Method
def fdm_1d_second_order(n, h=1.0):
    """
    Generate a 1D finite difference matrix for second-order central difference.

    Args:
    - n (int): Number of grid points.
    - h (float): Grid spacing.

    Returns:
    - numpy.ndarray: Finite difference matrix.
    """
    diagonals = [-2*np.ones(n), np.ones(n-1), np.ones(n-1)]
    return np.diag(diagonals[0]) + np.diag(diagonals[1], k=1) + np.diag(diagonals[1], k=-1) / h**2

# Finite Element Method
def fem_1d_linear_stiffness(n, h=1.0):
    """
    Generate a 1D finite element stiffness matrix using linear elements.

    Args:
    - n (int): Number of nodes.
    - h (float): Element length.

    Returns:
    - numpy.ndarray: Stiffness matrix.
    """
    elem_matrix = np.array([[1,-1], [-1,1]]) / h
    global_matrix = np.zeros((n,n))

    for i in range(n-1):
        global_matrix[i:i+2, i:i+2] += elem_matrix

    return global_matrix

# Matrix visualization
def plot_heatmap(matrix, title="Heatmap"):
    """Plot a heatmap for a matrix.

    Args:
    - matrix (numpy.ndarray): Matrix to visualize
    - title (str): Title for the plot    
    """
    plt.figure(figsize=(8,6))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()

def plot_spy(matrix, title="Spy Plot"):
    """Plot a spy plot for a matrix.
    
    Args:
    - matrix (numpy.ndarray): Matrix to visualize
    - title (str): Title for the plot    
    """
    plt.figure(figsize=(8, 6))
    plt.spy(matrix)
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()