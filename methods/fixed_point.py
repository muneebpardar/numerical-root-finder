import numpy as np

def fixed_point(f, g, x0, tol=1e-6, max_iter=100):
    """
    Fixed Point Iteration Method using original function f(x)
    
    Args:
        f: Original function f(x) = 0
        g: Iteration function g(x) where x = g(x)
        x0: Initial guess
        tol: Tolerance for convergence (relative error)
        max_iter: Maximum number of iterations
    
    Returns:
        dict with results including iterations table with relative error
    """
    iterations = []
    x_prev = x0
    
    # Calculate first iteration to get initial values
    try:
        x_new = g(x_prev)
        
        # Ensure we get real values only (handle complex results)
        if isinstance(x_new, complex) or np.iscomplexobj(x_new):
            x_new = np.real(x_new)
        if not np.isfinite(x_new) or np.isnan(x_new):
            raise ValueError(f"g(x) produced invalid value: {x_new}")
        
        # Calculate relative error: |xₙ₊₁ - xₙ| / |xₙ₊₁|
        if abs(x_new) > 1e-14:
            relative_error = abs(x_new - x_prev) / abs(x_new)
        else:
            # If x_new is very close to zero, use absolute error
            relative_error = abs(x_new - x_prev)
        
        # Store initial iteration (n=0)
        iterations.append({
            'n': 0,
            'xₙ': float(x_prev),
            'f(xₙ)': float(f(x_prev)),
            'xₙ₊₁': float(x_new),
            'Relative Error': float(relative_error)
        })
        
        # Check convergence on first iteration
        if relative_error < tol:
            return {
                'success': True,
                'converged': True,
                'root': x_new,
                'iterations': iterations,
                'message': f'Converged after 0 iterations'
            }
        
        x_prev = x_new
        
    except Exception as e:
        return {
            'success': False,
            'converged': False,
            'root': None,
            'iterations': [],
            'message': f'Error at initial iteration: {str(e)}'
        }
    
    # Continue iterations
    for n in range(1, max_iter + 1):
        try:
            # Calculate next value using g(x)
            x_new = g(x_prev)
            
            # Ensure we get real values only (handle complex results)
            if isinstance(x_new, complex) or np.iscomplexobj(x_new):
                x_new = np.real(x_new)
            if not np.isfinite(x_new) or np.isnan(x_new):
                raise ValueError(f"g(x) produced invalid value at iteration {n}: {x_new}")
            
            # Calculate relative error: |xₙ₊₁ - xₙ| / |xₙ₊₁|
            if abs(x_new) > 1e-14:
                relative_error = abs(x_new - x_prev) / abs(x_new)
            else:
                # If x_new is very close to zero, use absolute error
                relative_error = abs(x_new - x_prev)
            
            # Store iteration data (ensure all values are real floats)
            iterations.append({
                'n': n,
                'xₙ': float(x_prev),
                'f(xₙ)': float(f(x_prev)),
                'xₙ₊₁': float(x_new),
                'Relative Error': float(relative_error)
            })
            
            # Check convergence
            if relative_error < tol:
                return {
                    'success': True,
                    'converged': True,
                    'root': x_new,
                    'iterations': iterations,
                    'message': f'Converged after {n} iterations'
                }
            
            # Check for divergence
            if abs(x_new) > 1e10 or not np.isfinite(x_new):
                return {
                    'success': False,
                    'converged': False,
                    'root': None,
                    'iterations': iterations,
                    'message': f'Method diverged at iteration {n}'
                }
            
            x_prev = x_new
            
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            return {
                'success': False,
                'converged': False,
                'root': None,
                'iterations': iterations,
                'message': f'Error at iteration {n}: {str(e)}'
            }
    
    return {
        'success': True,
        'converged': False,
        'root': x_prev,
        'iterations': iterations,
        'message': f'Did not converge within {max_iter} iterations'
    }


def auto_transform_to_g(f_str, method='default'):
    """
    Automatically transform f(x) = 0 to x = g(x)
    
    Args:
        f_str: String representation of f(x)
        method: Transformation method
    
    Returns:
        String representation of g(x)
    """
    from sympy import symbols, sympify, solve, simplify
    from utils.validators import preprocess_function
    
    try:
        x = symbols('x')
        processed = preprocess_function(f_str)
        expr = sympify(processed)
        
        # Try to solve for x
        # f(x) = 0 → solve for x in terms of x
        # Example: x³ - x - 2 = 0 → x³ = x + 2 → x = (x + 2)^(1/3)
        
        # Method 1: Simple rearrangement
        # Move everything except one x term to the right
        g_expr = solve(expr, x)
        
        if g_expr and len(g_expr) > 0:
            # Use the first solution
            return str(simplify(g_expr[0]))
        
        # Method 2: Default transformation x = x - λf(x)
        # Use λ = 0.1 as default
        default_g = x - 0.1 * expr
        return str(simplify(default_g))
        
    except:
        # Fallback: return x - 0.1*f(x)
        return f"x - 0.1*({f_str})"


def check_convergence_condition(g_prime_func, a, b, num_points=100):
    """
    Check convergence condition |g'(x)| < 1 over interval [a, b]
    Only checks at extreme values (endpoints a and b) as per user requirement
    
    Args:
        g_prime_func: Derivative function g'(x)
        a: Left endpoint
        b: Right endpoint
        num_points: Number of points to check (for plotting, but convergence only at endpoints)
    
    Returns:
        dict with convergence info, where k is the highest value found
    """
    # Check convergence ONLY at endpoints (a and b)
    endpoint_vals = [a, b]
    g_prime_vals_at_endpoints = []
    
    for x in endpoint_vals:
        try:
            val = abs(float(g_prime_func(x)))
            if np.isfinite(val):
                g_prime_vals_at_endpoints.append(val)
            else:
                g_prime_vals_at_endpoints.append(np.inf)
        except:
            g_prime_vals_at_endpoints.append(np.inf)
    
    # k is the maximum value found at endpoints
    finite_endpoint_vals = [v for v in g_prime_vals_at_endpoints if v != np.inf and np.isfinite(v)]
    
    if len(finite_endpoint_vals) == 0:
        return {
            'converges': False,
            'max_derivative': np.inf,
            'min_derivative': np.inf,
            'x_values': endpoint_vals,
            'g_prime_values': g_prime_vals_at_endpoints,
            'k': np.inf
        }
    
    k = max(finite_endpoint_vals)  # Highest value found
    min_g_prime = min(finite_endpoint_vals)
    
    converges = k < 1
    
    # Also compute for plotting purposes (but not for convergence check)
    x_vals = np.linspace(a, b, num_points)
    g_prime_vals = []
    for x in x_vals:
        try:
            val = abs(float(g_prime_func(x)))
            if np.isfinite(val):
                g_prime_vals.append(val)
            else:
                g_prime_vals.append(np.inf)
        except:
            g_prime_vals.append(np.inf)
    
    # Find overall max for plotting (but k is from endpoints only)
    finite_vals = [v for v in g_prime_vals if v != np.inf and np.isfinite(v)]
    overall_max = max(finite_vals) if finite_vals else np.inf
    
    return {
        'converges': converges,
        'max_derivative': k,  # k is the max at endpoints
        'min_derivative': min_g_prime,
        'x_values': x_vals,  # For plotting
        'g_prime_values': g_prime_vals,  # For plotting
        'k': k  # Explicitly store k
    }


def calculate_max_iterations_formula(error, x0, a, b, k):
    """
    Calculate maximum iterations needed using the formula:
    n > (ln(Error) - ln[max(x0-a, b-x0)]) / ln(k)
    
    Where:
    - Error: desired error tolerance
    - x0: initial guess (first interval)
    - a: first interval endpoint
    - b: last interval endpoint
    - k: highest value found after convergence test (max |g'(x)| at endpoints)
    
    Args:
        error: Desired error tolerance
        x0: Initial guess
        a: Left interval endpoint
        b: Right interval endpoint
        k: Maximum |g'(x)| at endpoints (from convergence test)
    
    Returns:
        Estimated maximum iterations needed (n > result), or None if convergence is not guaranteed
    """
    try:
        # Check if convergence is guaranteed (k < 1)
        if k >= 1 or not np.isfinite(k) or k <= 0:
            return None  # Won't converge
        
        # Calculate max(x0-a, b-x0) - note: user said max(x0-a, b-x0), not absolute values
        # But typically we use absolute values, so using max(|x0-a|, |b-x0|)
        max_distance = max(abs(x0 - a), abs(b - x0))
        
        # Validate inputs
        if max_distance <= 0 or error <= 0 or not np.isfinite(max_distance) or not np.isfinite(error):
            return None
        
        # Calculate using the formula: n > (ln(Error) - ln[max(x0-a, b-x0)]) / ln(k)
        # Note: ln(k) is negative since k < 1, so we divide by negative number
        numerator = np.log(error) - np.log(max_distance)
        denominator = np.log(k)
        
        # Check if denominator is valid (should be negative since k < 1)
        if denominator >= 0 or not np.isfinite(denominator):
            return None  # Won't converge
        
        n = numerator / denominator
        
        # Since n > result, we return ceil(n) + 1 to ensure n > result
        # But typically we just return the minimum n that satisfies the inequality
        if n < 0:
            return None
        
        # Return the minimum integer n such that n > result
        return int(np.ceil(n)) + 1 if n > 0 else int(np.ceil(n))
        
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        return None