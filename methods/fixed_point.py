"""
Fixed Point Iteration Method Implementation
"""

import numpy as np


def fixed_point(g, x0, tol=1e-6, max_iter=100):
    """
    Find root using Fixed Point Iteration.
    
    Algorithm:
    1. Transform f(x) = 0 to x = g(x)
    2. Start with initial guess x₀
    3. Calculate: x₁ = g(x₀)
    4. If |x₁ - x₀| < tol, root found
    5. Set x₀ = x₁ and repeat
    
    Note: Convergence requires |g'(x)| < 1 near the root
    
    Args:
        g (function): The iteration function g(x)
        x0 (float): Initial guess
        tol (float): Tolerance for convergence
        max_iter (int): Maximum iterations
    
    Returns:
        dict: {
            'success': bool,
            'root': float,
            'iterations': list of dicts,
            'message': str
        }
    """
    
    iterations = []
    x = x0
    
    # Main iteration loop
    for n in range(1, max_iter + 1):
        # Apply iteration function
        x_new = g(x)
        
        # Calculate error
        error = abs(x_new - x)
        
        # Store iteration data
        iterations.append({
            'n': n,
            'xₙ': x,
            'g(xₙ)': x_new,
            'xₙ₊₁': x_new,
            'error': error
        })
        
        # Check convergence
        if error < tol:
            return {
                'success': True,
                'root': x_new,
                'iterations': iterations,
                'message': f'Root found after {n} iterations'
            }
        
        # Divergence check
        if abs(x_new) > 1e10:
            return {
                'success': False,
                'root': x_new,
                'iterations': iterations,
                'message': 'Method diverged - check if |g\'(x)| < 1'
            }
        
        # Update for next iteration
        x = x_new
    
    # Max iterations reached
    return {
        'success': False,
        'root': x_new,
        'iterations': iterations,
        'message': 'Maximum iterations reached'
    }