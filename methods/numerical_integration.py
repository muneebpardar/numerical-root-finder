"""
Numerical Integration Methods

This module implements three numerical integration methods:
1. Trapezoidal Rule
2. Simpson's 1/3 Rule
3. Simpson's 3/8 Rule

All methods approximate definite integrals ∫f(x)dx over interval [a, b] using discrete data points.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def validate_equally_spaced(x_points: List[float], tolerance: float = 1e-10) -> Tuple[bool, float, str]:
    """
    Validate that x points are equally spaced.
    
    Args:
        x_points: List of x values
        tolerance: Tolerance for floating point comparison
    
    Returns:
        Tuple of (is_valid, spacing_h, error_message)
    """
    if len(x_points) < 2:
        return False, 0.0, "Need at least 2 points"
    
    h = x_points[1] - x_points[0]
    
    for i in range(1, len(x_points)):
        expected = x_points[0] + i * h
        if abs(x_points[i] - expected) > tolerance:
            return False, h, f"Points not equally spaced. Expected {expected:.6f}, got {x_points[i]:.6f}"
    
    return True, h, ""


def validate_integration_input(x_points: List[float], y_points: List[float], method: str) -> Dict:
    """
    Validate inputs for numerical integration.
    
    Args:
        x_points: List of x coordinates
        y_points: List of y=f(x) values
        method: 'trapezoidal', 'simpson_1_3', or 'simpson_3_8'
    
    Returns:
        Dictionary with validation results
    """
    # Check equal lengths
    if len(x_points) != len(y_points):
        return {
            'valid': False,
            'message': 'x and y arrays must have same length',
            'h': None
        }
    
    # Check minimum points
    if len(x_points) < 2:
        return {
            'valid': False,
            'message': 'Need at least 2 points for integration',
            'h': None
        }
    
    # Check equally spaced
    is_valid, h, error_msg = validate_equally_spaced(x_points)
    if not is_valid:
        return {
            'valid': False,
            'message': f'Points must be equally spaced. {error_msg}',
            'h': h
        }
    
    # Method-specific validation
    n = len(x_points) - 1  # number of intervals
    
    if method == 'simpson_1_3':
        if n % 2 != 0:
            return {
                'valid': False,
                'message': f"Simpson's 1/3 requires EVEN number of intervals (ODD number of points). You have {len(x_points)} points (n={n} intervals).",
                'h': h
            }
    
    elif method == 'simpson_3_8':
        if n % 3 != 0:
            return {
                'valid': False,
                'message': f"Simpson's 3/8 requires number of intervals divisible by 3. You have {n} intervals (need 3, 6, 9, ...). Points needed: 4, 7, 10, 13, ...",
                'h': h
            }
    
    return {
        'valid': True,
        'message': 'Valid input',
        'h': h
    }


def trapezoidal_rule(x_points: List[float], y_points: List[float]) -> float:
    """
    Calculate integral using Trapezoidal Rule.
    
    Formula: ∫[x₀ to xₙ] f(x)dx = (h/2)[y₀ + 2y₁ + 2y₂ + ... + 2yₙ₋₁ + yₙ]
    
    Args:
        x_points: List of x coordinates (equally spaced)
        y_points: List of y=f(x) values
    
    Returns:
        Integral value
    """
    n = len(x_points) - 1
    h = x_points[1] - x_points[0]
    
    # First and last terms (coefficient 0.5 each, so total = 1)
    result = (y_points[0] + y_points[-1]) / 2.0
    
    # Middle terms (all have coefficient 1, but we already divided by 2)
    for i in range(1, n):
        result += y_points[i]
    
    result = result * h
    
    return result


def simpsons_one_third_rule(x_points: List[float], y_points: List[float]) -> float:
    """
    Calculate integral using Simpson's 1/3 Rule.
    
    Formula: ∫[x₀ to xₙ] f(x)dx = (h/3)[y₀ + 4y₁ + 2y₂ + 4y₃ + 2y₄ + ... + 4yₙ₋₁ + yₙ]
    
    Requires: EVEN number of intervals (ODD number of points)
    
    Args:
        x_points: List of x coordinates (equally spaced)
        y_points: List of y=f(x) values
    
    Returns:
        Integral value
    """
    n = len(x_points) - 1  # number of intervals
    h = x_points[1] - x_points[0]
    
    # First and last terms (coefficient 1)
    result = y_points[0] + y_points[-1]
    
    # Odd-indexed terms (coefficient 4)
    for i in range(1, n, 2):  # 1, 3, 5, 7, ...
        result += 4 * y_points[i]
    
    # Even-indexed terms (coefficient 2)
    for i in range(2, n, 2):  # 2, 4, 6, 8, ...
        result += 2 * y_points[i]
    
    result = (h / 3) * result
    
    return result


def simpsons_three_eighths_rule(x_points: List[float], y_points: List[float]) -> float:
    """
    Calculate integral using Simpson's 3/8 Rule.
    
    Formula: ∫[x₀ to xₙ] f(x)dx = (3h/8)[y₀ + 3y₁ + 3y₂ + 2y₃ + 3y₄ + 3y₅ + 2y₆ + ... + 3yₙ₋₁ + yₙ]
    
    Requires: Number of intervals divisible by 3
    
    Args:
        x_points: List of x coordinates (equally spaced)
        y_points: List of y=f(x) values
    
    Returns:
        Integral value
    """
    n = len(x_points) - 1  # number of intervals
    h = x_points[1] - x_points[0]
    
    # First and last terms (coefficient 1)
    result = y_points[0] + y_points[-1]
    
    # Process middle terms
    for i in range(1, n):
        if i % 3 == 0:
            # Boundary between groups (coefficient 2)
            result += 2 * y_points[i]
        else:
            # Within a group (coefficient 3)
            result += 3 * y_points[i]
    
    result = (3 * h / 8) * result
    
    return result


def calculate_formula_breakdown(x_points: List[float], y_points: List[float], method: str) -> Dict:
    """
    Calculate detailed breakdown of how formula is applied.
    
    Args:
        x_points: List of x coordinates
        y_points: List of y=f(x) values
        method: 'trapezoidal', 'simpson_1_3', or 'simpson_3_8'
    
    Returns:
        Dictionary with breakdown details
    """
    n = len(x_points) - 1
    h = x_points[1] - x_points[0]
    
    breakdown = {
        'h': h,
        'interval': (x_points[0], x_points[-1]),
        'interval_str': f"[{x_points[0]:.4f}, {x_points[-1]:.4f}]",
        'terms': []
    }
    
    if method == 'trapezoidal':
        # First term (coefficient 0.5)
        breakdown['terms'].append({
            'index': 0,
            'x': x_points[0],
            'y': y_points[0],
            'coefficient': 0.5,
            'contribution': 0.5 * y_points[0] * h
        })
        
        # Middle terms (coefficient 1.0)
        for i in range(1, n):
            breakdown['terms'].append({
                'index': i,
                'x': x_points[i],
                'y': y_points[i],
                'coefficient': 1.0,
                'contribution': y_points[i] * h
            })
        
        # Last term (coefficient 0.5)
        breakdown['terms'].append({
            'index': n,
            'x': x_points[n],
            'y': y_points[n],
            'coefficient': 0.5,
            'contribution': 0.5 * y_points[n] * h
        })
        
        breakdown['formula'] = f"(h/2)[y₀ + 2(y₁+...+y_{n-1}) + yₙ]"
        breakdown['formula_latex'] = r"\int_a^b f(x)dx \approx \frac{h}{2}[y_0 + 2y_1 + 2y_2 + \cdots + 2y_{n-1} + y_n]"
    
    elif method == 'simpson_1_3':
        for i in range(n + 1):
            if i == 0 or i == n:
                coeff = 1
            elif i % 2 == 1:  # odd index
                coeff = 4
            else:  # even index
                coeff = 2
            
            breakdown['terms'].append({
                'index': i,
                'x': x_points[i],
                'y': y_points[i],
                'coefficient': coeff,
                'contribution': coeff * y_points[i] * (h / 3)
            })
        
        breakdown['formula'] = f"(h/3)[y₀ + 4(odd terms) + 2(even terms) + yₙ]"
        breakdown['formula_latex'] = r"\int_a^b f(x)dx \approx \frac{h}{3}[y_0 + 4y_1 + 2y_2 + 4y_3 + 2y_4 + \cdots + 4y_{n-1} + y_n]"
    
    elif method == 'simpson_3_8':
        for i in range(n + 1):
            if i == 0 or i == n:
                coeff = 1
            elif i % 3 == 0:  # boundary between groups
                coeff = 2
            else:  # within a group
                coeff = 3
            
            breakdown['terms'].append({
                'index': i,
                'x': x_points[i],
                'y': y_points[i],
                'coefficient': coeff,
                'contribution': coeff * y_points[i] * (3 * h / 8)
            })
        
        breakdown['formula'] = f"(3h/8)[y₀ + 3(grouped terms) + 2(boundaries) + yₙ]"
        breakdown['formula_latex'] = r"\int_a^b f(x)dx \approx \frac{3h}{8}[y_0 + 3y_1 + 3y_2 + 2y_3 + 3y_4 + 3y_5 + 2y_6 + \cdots + 3y_{n-1} + y_n]"
    
    return breakdown


def numerical_integration(
    x_points: List[float],
    y_points: List[float],
    method: str = 'trapezoidal'
) -> Dict:
    """
    Perform numerical integration using specified method.
    
    Args:
        x_points: List/array of x coordinates (equally spaced)
        y_points: List/array of y=f(x) values
        method: 'trapezoidal', 'simpson_1_3', or 'simpson_3_8'
    
    Returns:
        Dictionary with results:
        {
            'success': bool,
            'integral': float,
            'method': str,
            'interval': tuple (a, b),
            'num_points': int,
            'num_intervals': int,
            'spacing_h': float,
            'formula_used': str,
            'breakdown': dict,
            'message': str
        }
    """
    # Convert to lists if numpy arrays
    x = list(x_points) if isinstance(x_points, np.ndarray) else x_points
    y = list(y_points) if isinstance(y_points, np.ndarray) else y_points
    
    # Validate input
    validation = validate_integration_input(x, y, method)
    if not validation['valid']:
        return {
            'success': False,
            'message': validation['message'],
            'integral': None
        }
    
    h = validation['h']
    n = len(x) - 1
    a, b = x[0], x[-1]
    
    # Calculate based on method
    try:
        if method == 'trapezoidal':
            integral = trapezoidal_rule(x, y)
            method_name = "Trapezoidal Rule"
            
        elif method == 'simpson_1_3':
            integral = simpsons_one_third_rule(x, y)
            method_name = "Simpson's 1/3 Rule"
            
        elif method == 'simpson_3_8':
            integral = simpsons_three_eighths_rule(x, y)
            method_name = "Simpson's 3/8 Rule"
            
        else:
            return {
                'success': False,
                'message': f'Unknown method: {method}',
                'integral': None
            }
        
        # Calculate breakdown
        breakdown = calculate_formula_breakdown(x, y, method)
        
        return {
            'success': True,
            'integral': integral,
            'method': method,
            'method_name': method_name,
            'interval': (a, b),
            'num_points': len(x),
            'num_intervals': n,
            'spacing_h': h,
            'formula_used': breakdown.get('formula', ''),
            'formula_latex': breakdown.get('formula_latex', ''),
            'breakdown': breakdown,
            'message': f'✅ Integration successful using {method_name}'
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Error during calculation: {str(e)}',
            'integral': None
        }
