import numpy as np

def jacobi_method(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Jacobi Iterative Method for solving linear systems Ax = b
    
    Parameters:
    -----------
    A : numpy.ndarray
        Coefficient matrix (must be square)
    b : numpy.ndarray
        Right-hand side vector
    x0 : numpy.ndarray, optional
        Initial guess (if None, uses zero vector)
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations
    
    Returns:
    --------
    dict with:
        - success: bool
        - message: str
        - solution: numpy.ndarray
        - iterations: list of iteration data
        - converged: bool
        - final_error: float
    """
    
    # Validate inputs
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    # Check if matrix is square
    if A.shape[0] != A.shape[1]:
        return {
            'success': False,
            'message': 'Matrix A must be square',
            'solution': None,
            'iterations': [],
            'converged': False,
            'final_error': None
        }
    
    n = A.shape[0]
    
    # Check if b has correct dimensions
    if b.shape[0] != n:
        return {
            'success': False,
            'message': 'Vector b must have same length as matrix A rows',
            'solution': None,
            'iterations': [],
            'converged': False,
            'final_error': None
        }
    
    # Check for zero diagonal elements
    if np.any(np.diag(A) == 0):
        return {
            'success': False,
            'message': 'Matrix A has zero diagonal elements. Jacobi method requires non-zero diagonal.',
            'solution': None,
            'iterations': [],
            'converged': False,
            'final_error': None
        }
    
    # Check for diagonal dominance (warning, not error)
    is_diagonally_dominant = True
    for i in range(n):
        row_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        if np.abs(A[i, i]) <= row_sum:
            is_diagonally_dominant = False
            break
    
    # Initial guess
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)
    
    # Store iterations
    iterations_data = []
    
    # Jacobi iteration
    converged = False
    
    for iteration in range(max_iter):
        x_new = np.zeros(n)
        
        # Jacobi formula: x_i^(k+1) = (b_i - sum(a_ij * x_j^(k))) / a_ii
        for i in range(n):
            sum_val = 0.0
            for j in range(n):
                if i != j:
                    sum_val += A[i, j] * x[j]
            
            x_new[i] = (b[i] - sum_val) / A[i, i]
        
        # Calculate error (infinity norm)
        error = np.linalg.norm(x_new - x, ord=np.inf)
        
        # Calculate residual (||Ax - b||)
        residual = np.linalg.norm(np.dot(A, x_new) - b, ord=np.inf)
        
        # Store iteration data
        iter_data = {
            'iteration': iteration + 1,
            'x': x_new.copy(),
            'error': error,
            'residual': residual
        }
        
        # Add individual components
        for i in range(n):
            iter_data[f'x{i+1}'] = x_new[i]
        
        iterations_data.append(iter_data)
        
        # Check convergence
        if error < tol:
            converged = True
            message = f'Converged after {iteration + 1} iterations'
            break
        
        # Update x
        x = x_new.copy()
    
    if not converged:
        message = f'Did not converge after {max_iter} iterations. Final error: {error:.2e}'
    
    # Add warning about diagonal dominance
    if not is_diagonally_dominant and converged:
        message += ' (Warning: Matrix is not strictly diagonally dominant, but method converged)'
    elif not is_diagonally_dominant and not converged:
        message += ' (Warning: Matrix is not strictly diagonally dominant, convergence not guaranteed)'
    
    return {
        'success': True,
        'message': message,
        'solution': x_new if converged else x,
        'iterations': iterations_data,
        'converged': converged,
        'final_error': error,
        'diagonally_dominant': is_diagonally_dominant
    }


def check_diagonal_dominance(A):
    """
    Check if matrix A is strictly diagonally dominant
    
    Parameters:
    -----------
    A : numpy.ndarray
        Coefficient matrix
    
    Returns:
    --------
    tuple: (is_dominant, details)
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    details = []
    is_dominant = True
    
    for i in range(n):
        diagonal = np.abs(A[i, i])
        row_sum = np.sum(np.abs(A[i, :])) - diagonal
        
        dominant_in_row = diagonal > row_sum
        
        details.append({
            'row': i + 1,
            'diagonal': diagonal,
            'off_diagonal_sum': row_sum,
            'is_dominant': dominant_in_row,
            'condition': f"|{A[i,i]:.10f}| > {row_sum:.10f}" if dominant_in_row else f"|{A[i,i]:.10f}| ≤ {row_sum:.10f}"
        })
        
        if not dominant_in_row:
            is_dominant = False
    
    return is_dominant, details


def format_matrix(A):
    """Format matrix for display"""
    if isinstance(A, list):
        A = np.array(A)
    
    n = A.shape[0]
    lines = []
    
    for i in range(n):
        row = "[ " + "  ".join([f"{A[i,j]:8.10f}" for j in range(A.shape[1])]) + " ]"
        lines.append(row)
    
    return "\n".join(lines)


def format_vector(v):
    """Format vector for display"""
    if isinstance(v, list):
        v = np.array(v)
    
    lines = []
    for i in range(len(v)):
        lines.append(f"[ {v[i]:8.10f} ]")
    
    return "\n".join(lines)