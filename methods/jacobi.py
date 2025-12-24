import numpy as np

def jacobi_method(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Jacobi Iterative Method for solving linear systems Ax = b
    
    The Jacobi method is an iterative algorithm that solves Ax = b by decomposing
    A into diagonal (D) and remainder (R) components: A = D + R
    
    Iteration formula: x^(k+1) = D^(-1)(b - Rx^(k))
    Or component-wise: x_i^(k+1) = (b_i - Σ(a_ij * x_j^(k), j≠i)) / a_ii
    
    Parameters:
    -----------
    A : numpy.ndarray or list
        Coefficient matrix (must be square, n×n)
    b : numpy.ndarray or list
        Right-hand side vector (length n)
    x0 : numpy.ndarray or list, optional
        Initial guess (if None, uses zero vector)
    tol : float, default=1e-6
        Convergence tolerance (infinity norm of error)
    max_iter : int, default=100
        Maximum number of iterations
    
    Returns:
    --------
    dict with keys:
        - success: bool - Whether method executed without errors
        - message: str - Status message
        - solution: numpy.ndarray - Final solution vector
        - iterations: list - Iteration history with x, error, residual
        - converged: bool - Whether method converged within tolerance
        - final_error: float - Final error value
        - diagonally_dominant: bool - Whether matrix is strictly diagonally dominant
    
    Notes:
    ------
    - Convergence is guaranteed if A is strictly diagonally dominant
    - Method may converge for some non-diagonally dominant matrices
    - Zero diagonal elements will cause failure
    """
    
    # Validate and convert inputs
    try:
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float).flatten()  # Ensure 1D
    except (ValueError, TypeError) as e:
        return {
            'success': False,
            'message': f'Invalid input: {str(e)}',
            'solution': None,
            'iterations': [],
            'converged': False,
            'final_error': None,
            'diagonally_dominant': False
        }
    
    # Check if matrix is square
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return {
            'success': False,
            'message': f'Matrix A must be square. Current shape: {A.shape}',
            'solution': None,
            'iterations': [],
            'converged': False,
            'final_error': None,
            'diagonally_dominant': False
        }
    
    n = A.shape[0]
    
    # Check if b has correct dimensions
    if b.shape[0] != n:
        return {
            'success': False,
            'message': f'Vector b length ({b.shape[0]}) must match matrix A rows ({n})',
            'solution': None,
            'iterations': [],
            'converged': False,
            'final_error': None,
            'diagonally_dominant': False
        }
    
    # Check for zero diagonal elements
    diagonal_elements = np.diag(A)
    if np.any(diagonal_elements == 0):
        zero_indices = np.where(diagonal_elements == 0)[0]
        return {
            'success': False,
            'message': f'Matrix has zero diagonal element(s) at row(s) {zero_indices + 1}. Jacobi method requires non-zero diagonal.',
            'solution': None,
            'iterations': [],
            'converged': False,
            'final_error': None,
            'diagonally_dominant': False
        }
    
    # Check for diagonal dominance (warning, not blocking)
    is_diagonally_dominant, dominance_details = check_diagonal_dominance(A)
    
    # Initial guess
    if x0 is None:
        x = np.zeros(n)
    else:
        try:
            x = np.array(x0, dtype=float).flatten()
            if x.shape[0] != n:
                return {
                    'success': False,
                    'message': f'Initial guess x0 length ({x.shape[0]}) must match system size ({n})',
                    'solution': None,
                    'iterations': [],
                    'converged': False,
                    'final_error': None,
                    'diagonally_dominant': is_diagonally_dominant
                }
        except (ValueError, TypeError) as e:
            return {
                'success': False,
                'message': f'Invalid initial guess: {str(e)}',
                'solution': None,
                'iterations': [],
                'converged': False,
                'final_error': None,
                'diagonally_dominant': is_diagonally_dominant
            }
    
    # Store iterations
    iterations_data = []
    
    # Store initial state (iteration 0)
    initial_residual = np.linalg.norm(np.dot(A, x) - b, ord=np.inf)
    iter_data_0 = {
        'iteration': 0,
        'x': x.copy(),
        'error': 0.0,  # No previous iteration to compare
        'residual': initial_residual
    }
    for i in range(n):
        iter_data_0[f'x{i+1}'] = x[i]
    iterations_data.append(iter_data_0)
    
    # Jacobi iteration
    converged = False
    error = 0.0
    
    for iteration in range(max_iter):
        x_new = np.zeros(n)
        
        # Jacobi formula: x_i^(k+1) = (b_i - sum(a_ij * x_j^(k) for j≠i)) / a_ii
        for i in range(n):
            sum_val = 0.0
            for j in range(n):
                if i != j:
                    sum_val += A[i, j] * x[j]
            
            x_new[i] = (b[i] - sum_val) / A[i, i]
        
        # Calculate relative error: ||x^(k) - x^(k-1)|| / ||x^(k)||
        numerator = np.linalg.norm(x_new - x, ord=np.inf)
        denominator = np.linalg.norm(x_new, ord=np.inf)

        if denominator < 1e-12:
            error = numerator
        else:
            error = numerator / denominator
        
        # Calculate residual (||Ax - b||_∞)
        residual = np.linalg.norm(np.dot(A, x_new) - b, ord=np.inf)
        
        # Store iteration data
        iter_data = {
            'iteration': iteration + 1,
            'x': x_new.copy(),
            'error': error,
            'residual': residual
        }
        
        # Add individual components for display
        for i in range(n):
            iter_data[f'x{i+1}'] = x_new[i]
        
        iterations_data.append(iter_data)
        
        # Check convergence
        if error < tol:
            converged = True
            message = f'✅ Converged after {iteration + 1} iterations'
            break
        
        # Check for divergence (optional safeguard)
        if error > 1e10:
            message = f'❌ Method diverged after {iteration + 1} iterations (error = {error:.2e})'
            break
        
        # Update x for next iteration
        x = x_new.copy()
    
    if not converged and error <= 1e10:
        message = f'⚠️ Did not converge after {max_iter} iterations. Final error: {error:.2e}'
    
    # Add warnings about diagonal dominance
    if not is_diagonally_dominant and converged:
        message += '\n⚠️ Note: Matrix is not strictly diagonally dominant, but method converged'
    elif not is_diagonally_dominant and not converged:
        message += '\n⚠️ Warning: Matrix is not strictly diagonally dominant. Convergence not guaranteed'
    
    return {
        'success': True,
        'message': message,
        'solution': x_new if converged else x,
        'iterations': iterations_data,
        'converged': converged,
        'final_error': error,
        'diagonally_dominant': is_diagonally_dominant,
        'dominance_details': dominance_details
    }


def check_diagonal_dominance(A):
    """
    Check if matrix A is strictly diagonally dominant
    
    A matrix is strictly diagonally dominant if for each row i:
    |a_ii| > Σ|a_ij| for all j ≠ i
    
    Parameters:
    -----------
    A : numpy.ndarray
        Coefficient matrix (n×n)
    
    Returns:
    --------
    tuple: (is_dominant: bool, details: list of dict)
        - is_dominant: True if all rows satisfy diagonal dominance
        - details: Per-row analysis with diagonal, off-diagonal sum, and status
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    details = []
    is_dominant = True
    
    for i in range(n):
        diagonal = np.abs(A[i, i])
        row_sum = np.sum(np.abs(A[i, :])) - diagonal
        
        # Strictly diagonally dominant: |a_ii| > sum (not >=)
        dominant_in_row = diagonal > row_sum
        
        details.append({
            'row': i + 1,
            'diagonal': diagonal,
            'off_diagonal_sum': row_sum,
            'is_dominant': dominant_in_row,
            'condition': f"|{A[i,i]:.4f}| {'>' if dominant_in_row else '≤'} {row_sum:.4f}",
            'status': '✅' if dominant_in_row else '❌'
        })
        
        if not dominant_in_row:
            is_dominant = False
    
    return is_dominant, details


def format_matrix(A, precision=4):
    """
    Format matrix for display
    
    Parameters:
    -----------
    A : numpy.ndarray or list
        Matrix to format
    precision : int
        Number of decimal places
    
    Returns:
    --------
    str: Formatted matrix string
    """
    if isinstance(A, list):
        A = np.array(A)
    
    n, m = A.shape
    lines = []
    
    # Find max width for alignment
    max_width = max(len(f"{A[i,j]:.{precision}f}") for i in range(n) for j in range(m))
    
    for i in range(n):
        row = "[ " + "  ".join([f"{A[i,j]:{max_width}.{precision}f}" for j in range(m)]) + " ]"
        lines.append(row)
    
    return "\n".join(lines)


def format_vector(v, precision=4):
    """
    Format vector for display
    
    Parameters:
    -----------
    v : numpy.ndarray or list
        Vector to format
    precision : int
        Number of decimal places
    
    Returns:
    --------
    str: Formatted vector string
    """
    if isinstance(v, list):
        v = np.array(v)
    
    v = v.flatten()
    n = len(v)
    
    # Find max width for alignment
    max_width = max(len(f"{v[i]:.{precision}f}") for i in range(n))
    
    lines = []
    for i in range(n):
        lines.append(f"[ {v[i]:{max_width}.{precision}f} ]")
    
    return "\n".join(lines)


# Additional utility function
def calculate_spectral_radius(A):
    """
    Calculate spectral radius of Jacobi iteration matrix
    ρ(T_J) = ρ(D^(-1)R) where A = D + R
    
    If ρ(T_J) < 1, Jacobi method converges
    
    Parameters:
    -----------
    A : numpy.ndarray
        Coefficient matrix
    
    Returns:
    --------
    float: Spectral radius
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    # D^(-1)
    D_inv = np.diag(1.0 / np.diag(A))
    
    # R = A - D
    D = np.diag(np.diag(A))
    R = A - D
    
    # Jacobi iteration matrix: T_J = -D^(-1) * R
    T_J = -np.dot(D_inv, R)
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(T_J)
    
    # Spectral radius is max absolute eigenvalue
    spectral_radius = np.max(np.abs(eigenvalues))
    
    return spectral_radius