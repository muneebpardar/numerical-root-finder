"""
Newton-Raphson Method Implementation
"""

import numpy as np


def newton_raphson(f, f_prime, x0, tol=1e-6, max_iter=100):
    """
    Find root using Newton-Raphson Method.
    
    Algorithm:
    1. Start with initial guess x₀
    2. Calculate: x₁ = x₀ - f(x₀)/f'(x₀)
    3. If |x₁ - x₀| < tol, root found
    4. Set x₀ = x₁ and repeat
    
    Args:
        f (function): The function f(x)
        f_prime (function): Derivative f'(x)
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
        fx = f(x)
        fpx = f_prime(x)
        
        # Check for zero derivative
        if abs(fpx) < 1e-14:
            return {
                'success': False,
                'root': x,
                'iterations': iterations,
                'message': f"Derivative near zero at iteration {n}"
            }
        
        # Newton's formula
        x_new = x - fx / fpx
        
        # Calculate error
        error = abs(x_new - x)
        
        # Store iteration data
        iterations.append({
            'n': n,
            'xₙ': x,
            'f(xₙ)': fx,
            "f'(xₙ)": fpx,
            'xₙ₊₁': x_new,
            'error': error
        })
        
        # Check convergence
        if error < tol or abs(fx) < tol:
            return {
                'success': True,
                'root': x_new,
                'iterations': iterations,
                'message': f'Root found after {n} iterations'
            }
        
        # Update for next iteration
        x = x_new
        
        # Divergence check
        if abs(x) > 1e10:
            return {
                'success': False,
                'root': x,
                'iterations': iterations,
                'message': 'Method diverged'
            }
    
    # Max iterations reached
    return {
        'success': False,
        'root': x,
        'iterations': iterations,
        'message': 'Maximum iterations reached'
    }