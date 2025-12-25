"""
Newton-Raphson Method Implementation with Domain Handling
"""

import numpy as np
import warnings


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
    
    # Suppress runtime warnings temporarily (we'll handle them)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        # Main iteration loop
        for n in range(1, max_iter + 1):
            try:
                fx = f(x)
                fpx = f_prime(x)
                
                # Check for invalid function values
                if not np.isfinite(fx):
                    return {
                        'success': False,
                        'root': x,
                        'iterations': iterations,
                        'message': f"Function undefined at x = {x:.6f} (iteration {n}). Try a different initial guess."
                    }
                
                # Check for invalid derivative values
                if not np.isfinite(fpx):
                    return {
                        'success': False,
                        'root': x,
                        'iterations': iterations,
                        'message': f"Derivative undefined at x = {x:.6f} (iteration {n}). Try a different initial guess."
                    }
                
                # Check for zero derivative
                if abs(fpx) < 1e-14:
                    return {
                        'success': False,
                        'root': x,
                        'iterations': iterations,
                        'message': f"Derivative near zero at x = {x:.6f} (iteration {n}). Method cannot proceed."
                    }
                
                # Newton's formula
                x_new = x - fx / fpx
                
                # Check if new value is finite
                if not np.isfinite(x_new):
                    return {
                        'success': False,
                        'root': x,
                        'iterations': iterations,
                        'message': f"Method produced invalid value at iteration {n}. Try a different initial guess."
                    }
                
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
                        'message': f'Method diverged (|x| > 10¹⁰). Try a different initial guess.'
                    }
                    
            except Exception as e:
                return {
                    'success': False,
                    'root': x,
                    'iterations': iterations,
                    'message': f'Error at iteration {n}: {str(e)}'
                }
        
        # Max iterations reached
        return {
            'success': False,
            'root': x,
            'iterations': iterations,
            'message': f'Maximum iterations reached. Last approximation: x ≈ {x:.6f}'
        }