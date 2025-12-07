"""
Input validation utilities for root-finding methods
"""

from sympy import symbols, sympify, lambdify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import numpy as np
import re


def preprocess_function(func_str):
    """
    Preprocess function string to handle common input errors.
    Adds implicit multiplication where needed.
    
    Examples:
        "2x" -> "2*x"
        "3cos(x)" -> "3*cos(x)"
        "xsin(x)" -> "x*sin(x)"
        "(x+1)(x-1)" -> "(x+1)*(x-1)"
    """
    # Remove extra spaces
    func_str = func_str.strip()
    
    # Add multiplication between number and x
    func_str = re.sub(r'(\d)([x])', r'\1*\2', func_str)
    
    # Add multiplication between x and number
    func_str = re.sub(r'([x])(\d)', r'\1*\2', func_str)
    
    # Add multiplication between number and function
    func_str = re.sub(r'(\d)(sin|cos|tan|exp|log|sqrt|abs)', r'\1*\2', func_str)
    
    # Add multiplication between x and function
    func_str = re.sub(r'([x])(sin|cos|tan|exp|log|sqrt|abs)', r'\1*\2', func_str)
    
    # Add multiplication between ) and (
    func_str = re.sub(r'\)\s*\(', r')*(', func_str)
    
    # Add multiplication between ) and number
    func_str = re.sub(r'\)(\d)', r')*\1', func_str)
    
    # Add multiplication between ) and x
    func_str = re.sub(r'\)([x])', r')*\1', func_str)
    
    # Add multiplication between number and (
    func_str = re.sub(r'(\d)\(', r'\1*(', func_str)
    
    # Add multiplication between x and (
    func_str = re.sub(r'([x])\(', r'\1*(', func_str)
    
    return func_str


def validate_function(func_str):
    """
    Validates if the input string is a valid mathematical function.
    
    Args:
        func_str (str): Function string like "x**3 - x - 2"
    
    Returns:
        tuple: (is_valid, f, error_message)
    """
    try:
        # Preprocess the function string
        processed_str = preprocess_function(func_str)
        
        x = symbols('x')
        
        # Try using sympy parser with implicit multiplication
        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(processed_str, transformations=transformations)
        
        f = lambdify(x, expr, 'numpy')
        
        # Test the function with a sample value
        test_val = f(1.0)
        
        if not np.isfinite(test_val):
            return False, None, "Function produces infinite/undefined values at x=1"
        
        return True, f, "Valid function"
    
    except SyntaxError as e:
        # Provide helpful error message
        return False, None, f"Syntax error: Check for missing operators (* for multiplication). Processed as: {processed_str}"
    
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