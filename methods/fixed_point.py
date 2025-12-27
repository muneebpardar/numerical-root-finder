import numpy as np

def check_convergence_condition(g_prime, a, b, num_samples=100):
    """
    Check if |g'(x)| < 1 over the interval [a, b]
    
    Args:
        g_prime: Derivative function of g(x)
        a, b: Interval bounds
        num_samples: Number of points to sample
    
    Returns:
        dict with convergence info
    """
    x_vals = np.linspace(a, b, num_samples)
    g_prime_vals = []
    
    for x in x_vals:
        try:
            val = abs(g_prime(x))
            if not np.isnan(val) and not np.isinf(val):
                g_prime_vals.append(val)
        except:
            pass
    
    if not g_prime_vals:
        return {'converges': False, 'max_derivative': float('inf')}
    
    max_derivative = max(g_prime_vals)
    
    return {
        'converges': max_derivative < 1,
        'max_derivative': max_derivative
    }


def calculate_max_iterations_formula(tolerance, x0, a, b, k):
    """
    Calculate estimated max iterations using the formula:
    n_max = ceil((ln(ε) - ln(max(|x0-a|, |b-x0|))) / ln(k))
    """
    if k >= 1:
        return None
    
    max_distance = max(abs(x0 - a), abs(b - x0))
    
    if max_distance <= 0 or tolerance <= 0:
        return None
    
    try:
        numerator = np.log(tolerance) - np.log(max_distance)
        denominator = np.log(k)
        
        if denominator >= 0:
            return None
        
        n_max = np.ceil(numerator / denominator)
        return int(n_max)
    except:
        return None


def fixed_point(f, g, x0, tol=1e-6, max_iter=100):
    """
    Fixed Point Iteration Method
    
    Args:
        f: Original function f(x) = 0 (for verification)
        g: Iteration function g(x) where x = g(x)
        x0: Initial guess
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
    
    Returns:
        dict with solution and iterations
    """
    iterations = []
    x_current = x0
    
    # Initial iteration (k=0)
    iterations.append({
        'n': 0,
        'xₙ': x_current,
        'g(xₙ)': g(x_current),
        'xₙ₊₁': g(x_current),
        'error': 0.0
    })
    
    converged = False
    
    for n in range(1, max_iter + 1):
        # Calculate next value
        try:
            x_next = g(x_current)
        except Exception as e:
            return {
                'success': False,
                'message': f'Error evaluating g(x): {str(e)}',
                'iterations': iterations,
                'root': None,
                'converged': False
            }
        
        # Check for NaN or Inf
        if np.isnan(x_next) or np.isinf(x_next):
            return {
                'success': False,
                'message': 'Iteration diverged (NaN or Inf encountered)',
                'iterations': iterations,
                'root': None,
                'converged': False
            }
        
        # Calculate error
        error = abs(x_next - x_current)
        
        # Store iteration
        iterations.append({
            'n': n,
            'xₙ': x_current,
            'g(xₙ)': x_next,
            'xₙ₊₁': x_next,
            'error': error
        })
        
        # Check convergence
        if error < tol:
            converged = True
            return {
                'success': True,
                'converged': True,
                'message': f'Converged after {n} iterations',
                'root': x_next,
                'iterations': iterations,
                'final_error': error
            }
        
        # Update for next iteration
        x_current = x_next
    
    # Max iterations reached
    return {
        'success': True,
        'converged': False,
        'message': f'Did not converge within {max_iter} iterations',
        'root': x_current,
        'iterations': iterations,
        'final_error': error
    }