"""
Input validation utilities for root-finding methods
"""

from sympy import symbols, sympify, lambdify
import numpy as np


def validate_function(func_str):
    """
    Validates if the input string is a valid mathematical function.
    
    Args:
        func_str (str): Function string like "x**3 - x - 2"
    
    Returns:
        tuple: (is_valid, f, error_message)
    """
    try:
        x = symbols('x')
        expr = sympify(func_str)
        f = lambdify(x, expr, 'numpy')
        
        # Test the function with a sample value
        test_val = f(1.0)
        
        if not np.isfinite(test_val):
            return False, None, "Function produces infinite/undefined values"
        
        return True, f, "Valid function"
    
    except Exception as e:
        return False, None, f"Invalid function: {str(e)}"


def validate_interval(f, a, b):
    """
    Validates interval for bisection/false position methods.
    Checks if f(a) and f(b) have opposite signs.
    
    Args:
        f (function): The function
        a (float): Left endpoint
        b (float): Right endpoint
    
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        if a >= b:
            return False, "Error: a must be less than b"
        
        fa = f(a)
        fb = f(b)
        
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return False, "Function undefined at interval endpoints"
        
        if fa * fb >= 0:
            return False, f"f(a) and f(b) must have opposite signs. f({a})={fa:.4f}, f({b})={fb:.4f}"
        
        return True, "Valid interval"
    
    except Exception as e:
        return False, f"Error evaluating function: {str(e)}"


def check_convergence(error, tolerance):
    """
    Check if error is below tolerance.
    """
    return abs(error) < tolerance