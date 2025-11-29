"""
Bisection Method Implementation
"""

import numpy as np


def bisection(f, a, b, tol=1e-6, max_iter=100):
    """
    Find root using Bisection Method.
    
    Algorithm:
    1. Start with interval [a, b] where f(a)*f(b) < 0
    2. Find midpoint c = (a + b) / 2
    3. If f(c) ≈ 0, root found
    4. If f(a)*f(c) < 0, root in [a, c], set b = c
    5. Else root in [c, b], set a = c
    6. Repeat until error < tolerance
    
    Args:
        f (function): The function f(x)
        a (float): Left endpoint
        b (float): Right endpoint
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
    
    # Check initial condition
    fa = f(a)
    fb = f(b)
    
    if fa * fb >= 0:
        return {
            'success': False,
            'root': None,
            'iterations': [],
            'message': 'f(a) and f(b) must have opposite signs'
        }
    
    # Main iteration loop
    for n in range(1, max_iter + 1):
        # Calculate midpoint
        c = (a + b) / 2
        fc = f(c)
        
        # Calculate error (half of interval width)
        error = abs(b - a) / 2
        
        # Store iteration data
        iterations.append({
            'n': n,
            'a': a,
            'b': b,
            'c': c,
            'f(c)': fc,
            'error': error
        })
        
        # Check convergence
        if error < tol or abs(fc) < tol:
            return {
                'success': True,
                'root': c,
                'iterations': iterations,
                'message': f'Root found after {n} iterations'
            }
        
        # Update interval
        if f(a) * fc < 0:
            b = c  # Root in left half
        else:
            a = c  # Root in right half
    
    # Max iterations reached
    return {
        'success': False,
        'root': c,
        'iterations': iterations,
        'message': 'Maximum iterations reached'
    }