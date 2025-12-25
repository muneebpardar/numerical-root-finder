"""
Fixed validators.py - handles function calls correctly
Save as: utils/validators.py
"""

import numpy as np
from sympy import symbols, sympify, lambdify
from sympy.functions import asin, acos, atan, sin, cos, tan, sqrt, exp, log, Abs
from sympy.core.numbers import pi as sympy_pi
import re

def preprocess_function(func_str):
    """
    Preprocess function string to handle mathematical notations
    
    CRITICAL: Must NOT add * between function names and their parentheses!
    """
    
    # Remove all whitespace
    func_str = func_str.replace(' ', '')
    
    # Handle inverse trig notation FIRST
    func_str = re.sub(r'sin\^-1', 'asin', func_str, flags=re.IGNORECASE)
    func_str = re.sub(r'cos\^-1', 'acos', func_str, flags=re.IGNORECASE)
    func_str = re.sub(r'tan\^-1', 'atan', func_str, flags=re.IGNORECASE)
    func_str = func_str.replace('arcsin', 'asin')
    func_str = func_str.replace('arccos', 'acos')
    func_str = func_str.replace('arctan', 'atan')
    
    # Replace ^ with ** for powers
    func_str = re.sub(r'\^', '**', func_str)
    
    # IMPLICIT MULTIPLICATION - CAREFUL ORDER MATTERS!
    
    # 1. Number * pi: 0.5pi → 0.5*pi
    func_str = re.sub(r'(\d\.?\d*)(pi)', r'\1*\2', func_str, flags=re.IGNORECASE)
    
    # 2. Number * variable (but NOT if followed by known functions)
    #    2x → 2*x, but NOT 2sin → 2*sin (we'll handle that separately)
    func_str = re.sub(r'(\d)([a-z])(?![a-z])', r'\1*\2', func_str, flags=re.IGNORECASE)
    
    # 3. Number * function: 2sin(x) → 2*sin(x), 10sqrt(x) → 10*sqrt(x)
    #    BUT: Do NOT add * before the opening parenthesis of the function!
    #    Match: digit followed by function name (not followed by *)
    func_str = re.sub(r'(\d)(asin|acos|atan|sin|cos|tan|sqrt|exp|log|abs)(?!\*)', r'\1*\2', func_str, flags=re.IGNORECASE)
    
    # 4. ) * (: )( → )*(
    func_str = re.sub(r'\)\(', ')*(', func_str)
    
    # 5. ) * number: )2 → )*2
    func_str = re.sub(r'\)(\d)', r')*\1', func_str)
    
    # 6. Number * (: 2( → 2*( BUT only if not preceded by a function name
    #    This is for cases like: 2(x+1) → 2*(x+1)
    #    But NOT: sin(x) → sin*(x)
    func_str = re.sub(r'(\d)\((?![a-z])', r'\1*(', func_str)
    
    # 7. ) * variable: )x → )*x
    func_str = re.sub(r'\)([a-z])(?![a-z])', r')*\1', func_str, flags=re.IGNORECASE)
    
    # 8. Variable * (: x( → x*(  BUT ONLY for single variables, NOT function names
    #    This regex uses negative lookahead to exclude function names
    #    It will match: x(, y(, z(  but NOT: asin(, sin(, sqrt(, etc.
    func_str = re.sub(
        r'(?<![a-z])([a-z])(?!sin|cos|tan|sqrt|exp|log|abs|pi)\(', 
        r'\1*(', 
        func_str, 
        flags=re.IGNORECASE
    )
    
    return func_str


def validate_function(func_str):
    """
    Validate and parse mathematical function string
    
    Returns:
    --------
    tuple: (is_valid: bool, function: callable or None, error_message: str or None)
    """
    
    if not func_str or func_str.strip() == "":
        return False, None, "Function string is empty"
    
    try:
        # Preprocess the function
        processed_str = preprocess_function(func_str)
        
        # Create symbol
        x = symbols('x')
        
        # Create namespace with all sympy functions
        namespace = {
            'x': x,
            'asin': asin,
            'acos': acos,
            'atan': atan,
            'sin': sin,
            'cos': cos,
            'tan': tan,
            'sqrt': sqrt,
            'exp': exp,
            'log': log,
            'Abs': Abs,
            'abs': Abs,
            'pi': sympy_pi
        }
        
        # Parse using sympify with namespace
        expr = sympify(processed_str, locals=namespace)
        
        # Convert to numpy-compatible function
        numpy_funcs = {
            'asin': np.arcsin,
            'acos': np.arccos,
            'atan': np.arctan,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'sqrt': np.sqrt,
            'exp': np.exp,
            'log': np.log,
            'Abs': np.abs,
            'pi': np.pi
        }
        
        f = lambdify(x, expr, modules=[numpy_funcs, 'numpy'])
        
        # Test the function with safe values
        try:
            # Test with a value in domain of most functions
            test_result = f(0.5)
            if not np.isfinite(test_result):
                # Try zero
                test_result = f(0.0)
                if not np.isfinite(test_result):
                    return False, None, "Function produces non-finite values at test points"
        except Exception as e:
            # Function might have restricted domain, that's okay
            # As long as it parses correctly
            pass
        
        return True, f, None
        
    except SyntaxError as e:
        return False, None, f"Syntax error: {str(e)}"
    except TypeError as e:
        return False, None, f"Type error (check function syntax): {str(e)}"
    except Exception as e:
        return False, None, f"Error parsing function: {str(e)}"


def validate_interval(f, a, b):
    """
    Validate interval [a, b] for root-finding methods
    
    Checks:
    - a < b
    - f(a) and f(b) have opposite signs (for bisection/false position)
    """
    
    if a >= b:
        return False, f"Invalid interval: a ({a}) must be less than b ({b})"
    
    try:
        fa = f(a)
        fb = f(b)
        
        if np.isnan(fa) or np.isinf(fa):
            return False, f"f(a) = f({a}) is NaN or infinity"
        
        if np.isnan(fb) or np.isinf(fb):
            return False, f"f(b) = f({b}) is NaN or infinity"
        
        # Check for sign change
        if fa * fb > 0:
            return False, f"f(a) and f(b) must have opposite signs. f({a:.4f}) = {fa:.4f}, f({b:.4f}) = {fb:.4f}"
        
        return True, "Valid interval with sign change"
        
    except Exception as e:
        return False, f"Error evaluating function at interval endpoints: {str(e)}"