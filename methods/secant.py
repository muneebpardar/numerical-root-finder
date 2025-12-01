"""
Secant Method Implementation
"""

import numpy as np


def secant(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Find root using Secant Method.
    
    Algorithm:
    1. Start with two initial guesses x₀ and x₁
    2. Calculate: x₂ = x₁ - f(x₁) * (x₁ - x₀) / (f(x₁) - f(x₀))
    3. If |x₂ - x₁| < tol, root found
    4. Set x₀ = x₁, x₁ = x₂ and repeat
    
    Args:
        f (function): The function f(x)
        x0 (float): First initial guess
        x1 (float): Second initial guess
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
    
    # Main iteration loop
    for n in range(1, max_iter + 1):
        fx0 = f(x0)
        fx1 = f(x1)
        
        # Check for division by zero
        if abs(fx1 - fx0) < 1e-14:
            return {
                'success': False,
                'root': x1,
                'iterations': iterations,
                'message': f'Division by zero at iteration {n}'
            }
        
        # Secant formula
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        
        # Calculate error
        error = abs(x2 - x1)
        
        # Store iteration data
        iterations.append({
            'n': n,
            'x₀': x0,
            'x₁': x1,
            'f(x₀)': fx0,
            'f(x₁)': fx1,
            'x₂': x2,
            'error': error
        })
        
        # Check convergence
        if error < tol or abs(f(x2)) < tol:
            return {
                'success': True,
                'root': x2,
                'iterations': iterations,
                'message': f'Root found after {n} iterations'
            }
        
        # Update for next iteration
        x0 = x1
        x1 = x2
        
        # Divergence check
        if abs(x2) > 1e10:
            return {
                'success': False,
                'root': x2,
                'iterations': iterations,
                'message': 'Method diverged'
            }
    
    # Max iterations reached
    return {
        'success': False,
        'root': x2,
        'iterations': iterations,
        'message': 'Maximum iterations reached'
    }