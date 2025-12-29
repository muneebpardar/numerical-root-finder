"""
Newton-Raphson Method Implementation with Domain Handling
"""

import numpy as np
import warnings


def ivp_test(f, f_prime, x0, tolerance=1e-6, a=None, b=None):
    """
    Initial Value Problem (IVP) Test for Newton-Raphson Method.
    
    Tests if Newton's method is likely to converge from the given initial guess x₀.
    
    Conditions checked:
    1. f'(x₀) is not zero or too close to zero
    2. f(x₀) is finite and well-defined
    3. f'(x₀) is finite and well-defined
    4. The ratio |f(x₀)/f'(x₀)| is reasonable (not too large)
    5. The second derivative condition (optional convergence check)
    6. Sign change check in interval [a, b] (if provided) - like bisection method
    
    Args:
        f: The function f(x)
        f_prime: The derivative f'(x)
        x0: Initial guess
        tolerance: Tolerance for convergence checks
        a: Optional lower bound of interval
        b: Optional upper bound of interval
    
    Returns:
        dict: {
            'pass': bool,
            'message': str,
            'details': dict with test results
        }
    """
    details = {}
    warnings_list = []
    errors_list = []
    
    try:
        # Test 1: Check if f(x₀) is defined
        try:
            fx0 = f(x0)
            if not np.isfinite(fx0):
                errors_list.append(f"f(x₀) is not finite at x₀ = {x0}")
                return {
                    'pass': False,
                    'message': f"❌ IVP Test Failed: f(x₀) is not finite at x₀ = {x0}",
                    'details': {'f_x0': fx0, 'errors': errors_list}
                }
            details['f_x0'] = fx0
        except Exception as e:
            errors_list.append(f"f(x₀) cannot be evaluated: {str(e)}")
            return {
                'pass': False,
                'message': f"❌ IVP Test Failed: Cannot evaluate f(x₀) at x₀ = {x0}",
                'details': {'errors': errors_list}
            }
        
        # Test 2: Check if f'(x₀) is defined and not zero
        try:
            fpx0 = f_prime(x0)
            if not np.isfinite(fpx0):
                errors_list.append(f"f'(x₀) is not finite at x₀ = {x0}")
                return {
                    'pass': False,
                    'message': f"❌ IVP Test Failed: f'(x₀) is not finite at x₀ = {x0}",
                    'details': {'f_prime_x0': fpx0, 'errors': errors_list}
                }
            details['f_prime_x0'] = fpx0
            
            # Check if derivative is too close to zero
            if abs(fpx0) < 1e-14:
                errors_list.append(f"f'(x₀) is too close to zero: |f'(x₀)| = {abs(fpx0):.2e}")
                return {
                    'pass': False,
                    'message': f"❌ IVP Test Failed: f'(x₀) ≈ 0 at x₀ = {x0}. Newton's method cannot proceed.",
                    'details': {'f_prime_x0': fpx0, 'errors': errors_list}
                }
            
            # Warning if derivative is small
            if abs(fpx0) < 1e-6:
                warnings_list.append(f"f'(x₀) is small: |f'(x₀)| = {abs(fpx0):.2e}. Method may converge slowly.")
        except Exception as e:
            errors_list.append(f"f'(x₀) cannot be evaluated: {str(e)}")
            return {
                'pass': False,
                'message': f"❌ IVP Test Failed: Cannot evaluate f'(x₀) at x₀ = {x0}",
                'details': {'errors': errors_list}
            }
        
        # Test 3: Check the step size |f(x₀)/f'(x₀)|
        try:
            step_size = abs(fx0 / fpx0) if fpx0 != 0 else float('inf')
            details['step_size'] = step_size
            
            if step_size > 1e10:
                warnings_list.append(f"Initial step size is very large: {step_size:.2e}. Method may diverge.")
            elif step_size > 100:
                warnings_list.append(f"Initial step size is large: {step_size:.2e}. Method may converge slowly.")
        except:
            pass
        
        # Test 4: Check if x₁ = x₀ - f(x₀)/f'(x₀) is reasonable
        try:
            x1 = x0 - fx0 / fpx0
            if not np.isfinite(x1):
                errors_list.append(f"x₁ = {x1} is not finite")
                return {
                    'pass': False,
                    'message': f"❌ IVP Test Failed: First iteration produces invalid value",
                    'details': {'x1': x1, 'errors': errors_list}
                }
            details['x1'] = x1
            details['x1_distance'] = abs(x1 - x0)
        except Exception as e:
            errors_list.append(f"Cannot compute x₁: {str(e)}")
        
        # Test 5: Check if f(x₀) is close to zero (already near root)
        if abs(fx0) < tolerance:
            warnings_list.append(f"f(x₀) = {fx0:.2e} is already very close to zero. x₀ may already be a root.")
        
        # Test 6: Sign change check in interval [a, b] (if provided)
        sign_change_check = None
        if a is not None and b is not None:
            try:
                if a >= b:
                    warnings_list.append(f"Invalid interval: a ({a}) must be less than b ({b})")
                else:
                    fa = f(a)
                    fb = f(b)
                    
                    if not (np.isfinite(fa) and np.isfinite(fb)):
                        warnings_list.append(f"Cannot evaluate function at interval endpoints")
                    else:
                        # Check for sign change (like bisection method)
                        sign_change = (fa * fb) < 0
                        sign_change_check = {
                            'a': a,
                            'b': b,
                            'f(a)': fa,
                            'f(b)': fb,
                            'sign_change': sign_change
                        }
                        details['sign_change_check'] = sign_change_check
                        
                        if sign_change:
                            # Sign change detected - IVP test satisfied (root exists in interval)
                            return {
                                'pass': True,
                                'message': f"✅ IVP Test Passed: Sign change detected in interval [{a}, {b}]. Root exists in this interval (like bisection method).",
                                'details': details,
                                'warnings': warnings_list,
                                'sign_change_detected': True
                            }
                        else:
                            warnings_list.append(f"No sign change in interval [{a}, {b}]. f(a) = {fa:.6f}, f(b) = {fb:.6f}. Root may not exist in this interval.")
            except Exception as e:
                warnings_list.append(f"Error checking sign change: {str(e)}")
        
        # Determine overall result
        if errors_list:
            return {
                'pass': False,
                'message': f"❌ IVP Test Failed: {'; '.join(errors_list)}",
                'details': details,
                'warnings': warnings_list
            }
        elif warnings_list:
            return {
                'pass': True,
                'message': f"⚠️ IVP Test Passed with warnings: {'; '.join(warnings_list)}",
                'details': details,
                'warnings': warnings_list
            }
        else:
            return {
                'pass': True,
                'message': f"✅ IVP Test Passed: Initial conditions are suitable for Newton's method",
                'details': details,
                'warnings': []
            }
            
    except Exception as e:
        return {
            'pass': False,
            'message': f"❌ IVP Test Error: {str(e)}",
            'details': {'errors': [str(e)]}
        }


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
                
                # Newton's formula: xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)
                x_new = x - fx / fpx
                
                # Check if new value is finite
                if not np.isfinite(x_new):
                    return {
                        'success': False,
                        'root': x,
                        'iterations': iterations,
                        'message': f"Method produced invalid value at iteration {n}. Try a different initial guess."
                    }
                
                # Calculate relative error: (xₙ₊₁ - xₙ) / xₙ₊₁
                if abs(x_new) > 1e-14:  # Avoid division by zero
                    error = abs((x_new - x) / x_new)
                else:
                    # If x_new is very close to zero, use absolute error
                    error = abs(x_new - x)
                
                # Store iteration data (no step column)
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