"""
False Position (Regula Falsi) Method Implementation
"""

import numpy as np


def false_position(f, a, b, tol=1e-6, max_iter=100):
    """
    Find root using False Position Method.
    
    Algorithm:
    1. Start with interval [a, b] where f(a)*f(b) < 0
    2. Find weighted point: c = (a*f(b) - b*f(a)) / (f(b) - f(a))
    3. Evaluate f(c)
    4. If |f(c)| < tol, root found
    5. If f(a)*f(c) < 0, root in [a, c], set b = c
    6. Else root in [c, b], set a = c
    7. Repeat until convergence
    
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
    
    c_prev = a  # For error calculation
    
    # Main iteration loop
    for n in range(1, max_iter + 1):
        # Calculate false position point
        fa = f(a)
        fb = f(b)
        
        # Avoid division by zero
        if abs(fb - fa) < 1e-14:
            return {
                'success': False,
                'root': c_prev,
                'iterations': iterations,
                'message': 'Division by zero encountered'
            }
        
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        
        # Calculate error
        if n == 1:
            error = abs(b - a)
        else:
            error = abs(c - c_prev)
        
        # Store iteration data
        iterations.append({
            'n': n,
            'a': a,
            'b': b,
            'c': c,
            'f(a)': fa,
            'f(b)': fb,
            'f(c)': fc,
            'error': error
        })
        
        # Check convergence
        if abs(fc) < tol or error < tol:
            return {
                'success': True,
                'root': c,
                'iterations': iterations,
                'message': f'Root found after {n} iterations'
            }
        
        # Update interval
        if fa * fc < 0:
            b = c
        else:
            a = c
        
        c_prev = c
    
    # Max iterations reached
    return {
        'success': False,
        'root': c,
        'iterations': iterations,
        'message': 'Maximum iterations reached'
    }