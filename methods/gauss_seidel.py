"""
Gauss-Seidel Method for solving linear systems Ax = b
Save this as: methods/gauss_seidel.py
"""

import numpy as np

def gauss_seidel_method(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Solve linear system Ax = b using Gauss-Seidel iterative method
    
    Parameters:
    -----------
    A : numpy.ndarray
        Coefficient matrix (n x n)
    b : numpy.ndarray
        Right-hand side vector (n x 1)
    x0 : numpy.ndarray, optional
        Initial guess (default: zero vector)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations
    
    Returns:
    --------
    dict : Dictionary containing:
        - success: bool
        - converged: bool
        - message: str
        - solution: numpy.ndarray
        - iterations: list of iteration data
        - final_error: float
    """
    
    n = len(b)
    
    # Initialize with zero vector if not provided
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)
    
    # Check if diagonal elements are non-zero
    if np.any(np.diag(A) == 0):
        return {
            'success': False,
            'converged': False,
            'message': 'Matrix has zero diagonal elements',
            'solution': None,
            'iterations': [],
            'final_error': None
        }
    
    iterations = []
    
    # Store initial iteration
    initial_residual = np.linalg.norm(np.dot(A, x) - b, ord=np.inf)
    iter_data = {
        'iteration': 0,
        'x': x.copy()
    }
    
    # Add individual components
    for i in range(n):
        iter_data[f'x{i+1}'] = x[i]
    
    iter_data['error'] = 0.0
    iter_data['residual'] = initial_residual
    
    iterations.append(iter_data)
    
    # Main iteration loop
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        
        # Gauss-Seidel update: use updated values immediately
        for i in range(n):
            sum_val = 0.0
            
            # Use updated values for j < i (already computed in this iteration)
            for j in range(i):
                sum_val += A[i, j] * x[j]
            
            # Use old values for j > i (not yet computed in this iteration)
            for j in range(i + 1, n):
                sum_val += A[i, j] * x_old[j]
            
            x[i] = (b[i] - sum_val) / A[i, i]
        
        # Calculate error using formula: |(x1^k - x1^(k-1)) / x2^k|
        # x1 is the first component (index 0), x2 is the second component (index 1)
        if abs(x[1]) < 1e-12:
            # If x2 is too close to zero, use a fallback
            error = abs(x[0] - x_old[0])
        else:
            error = abs((x[0] - x_old[0]) / x[1])

        residual = np.linalg.norm(np.dot(A, x) - b, ord=np.inf)
        
        # Store iteration data
        iter_data = {
            'iteration': k,
            'x': x.copy()
        }
        
        # Add individual components
        for i in range(n):
            iter_data[f'x{i+1}'] = x[i]
        
        iter_data['error'] = error
        iter_data['residual'] = residual
        
        iterations.append(iter_data)
        
        # Check convergence
        if error < tol:
            return {
                'success': True,
                'converged': True,
                'message': f'✅ Converged in {k} iterations',
                'solution': x,
                'iterations': iterations,
                'final_error': error
            }
    
    # Max iterations reached
    return {
        'success': True,
        'converged': False,
        'message': f'⚠️ Maximum iterations ({max_iter}) reached without convergence',
        'solution': x,
        'iterations': iterations,
        'final_error': error
    }


def calculate_iteration_matrix_gs(A):
    """
    Calculate the Gauss-Seidel iteration matrix T_GS = -(D+L)^(-1)U
    where A = D + L + U (diagonal + lower + upper)
    
    Parameters:
    -----------
    A : numpy.ndarray
        Coefficient matrix
    
    Returns:
    --------
    T_GS : numpy.ndarray
        Gauss-Seidel iteration matrix
    """
    n = len(A)
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    
    try:
        D_plus_L_inv = np.linalg.inv(D + L)
        T_GS = -np.dot(D_plus_L_inv, U)
        return T_GS
    except:
        return None


def calculate_spectral_radius_gs(A):
    """
    Calculate spectral radius of Gauss-Seidel iteration matrix
    
    Parameters:
    -----------
    A : numpy.ndarray
        Coefficient matrix
    
    Returns:
    --------
    float : Spectral radius ρ(T_GS)
    """
    T_GS = calculate_iteration_matrix_gs(A)
    if T_GS is None:
        return None
    
    eigenvalues = np.linalg.eigvals(T_GS)
    return np.max(np.abs(eigenvalues))