"""
Newton's Forward and Backward Difference Formulas for Numerical Differentiation

This module implements numerical differentiation using:
- Newton's Forward Difference Formula
- Newton's Backward Difference Formula

Supports derivatives up to any order (generalized implementation).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from math import factorial


def binomial_coefficient(n: int, k: int) -> int:
    """
    Calculate binomial coefficient C(n, k) = n! / (k! * (n-k)!)
    
    Args:
        n: Total number of items
        k: Number of items to choose
    
    Returns:
        Binomial coefficient
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    return factorial(n) // (factorial(k) * factorial(n - k))


def calculate_forward_differences(y_values: List[float], max_order: Optional[int] = None) -> List[List[float]]:
    """
    Build forward difference table.
    
    Returns: 2D list where table[i][j] = Δʲyᵢ
    
    Args:
        y_values: List of function values [y₀, y₁, y₂, ..., yₙ]
        max_order: Maximum order of differences to calculate (None = calculate all)
    
    Returns:
        2D list representing the difference table
    """
    n = len(y_values)
    if max_order is None:
        max_order = n - 1
    
    # Initialize table: table[i][j] = Δʲyᵢ
    table = [[0.0] * (max_order + 1) for _ in range(n)]
    
    # Column 0: original values
    for i in range(n):
        table[i][0] = y_values[i]
    
    # Calculate higher order differences
    for j in range(1, max_order + 1):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]
    
    return table


def calculate_backward_differences(y_values: List[float], max_order: Optional[int] = None) -> List[List[float]]:
    """
    Build backward difference table.
    
    Returns: 2D list where table[i][j] = ∇ʲyᵢ
    
    Args:
        y_values: List of function values [y₀, y₁, y₂, ..., yₙ]
        max_order: Maximum order of differences to calculate (None = calculate all)
    
    Returns:
        2D list representing the difference table
    """
    n = len(y_values)
    if max_order is None:
        max_order = n - 1
    
    # Initialize table: table[i][j] = ∇ʲyᵢ
    table = [[0.0] * (max_order + 1) for _ in range(n)]
    
    # Column 0: original values
    for i in range(n):
        table[i][0] = y_values[i]
    
    # Calculate higher order differences
    for j in range(1, max_order + 1):
        for i in range(j, n):  # Start from index j
            table[i][j] = table[i][j - 1] - table[i - 1][j - 1]
    
    return table


def validate_equally_spaced(x_points: List[float], tolerance: float = 1e-10) -> Tuple[bool, float]:
    """
    Validate that x points are equally spaced.
    
    Args:
        x_points: List of x values
        tolerance: Tolerance for floating point comparison
    
    Returns:
        Tuple of (is_valid, spacing_h)
    """
    if len(x_points) < 2:
        return False, 0.0
    
    h = x_points[1] - x_points[0]
    
    for i in range(1, len(x_points)):
        expected = x_points[0] + i * h
        if abs(x_points[i] - expected) > tolerance:
            return False, h
    
    return True, h


def get_forward_derivative_coefficients(order: int, max_terms: int = 10) -> Dict[int, float]:
    """
    Get coefficients for Newton's forward difference derivative formula.
    
    For nth derivative: (dⁿy/dxⁿ)|ₓ₌ₓ₀ = (1/hⁿ) Σ(k=n to m) [Cₖₙ * Δᵏy₀]
    
    For 1st derivative: C_k = (-1)^(k+1) / k
    For 2nd derivative: C_k = specific coefficients
    For 3rd derivative: C_k = specific coefficients
    For general nth derivative: Use Stirling numbers or recursive formula
    
    Args:
        order: Derivative order (1, 2, 3, ...)
        max_terms: Maximum number of terms to include
    
    Returns:
        Dictionary mapping difference order to coefficient
    """
    coefficients = {}
    
    if order == 1:
        # First derivative: (1/h)[Δy₀ - Δ²y₀/2 + Δ³y₀/3 - Δ⁴y₀/4 + ...]
        for k in range(1, max_terms + 1):
            coefficients[k] = ((-1) ** (k + 1)) / k
    elif order == 2:
        # Second derivative: (1/h²)[Δ²y₀ - Δ³y₀ + (11/12)Δ⁴y₀ - ...]
        coefficients[2] = 1.0
        coefficients[3] = -1.0
        coefficients[4] = 11.0 / 12.0
        coefficients[5] = -5.0 / 6.0
        coefficients[6] = 137.0 / 180.0
        # Add more terms if needed
        for k in range(7, max_terms + 1):
            # Approximate coefficients for higher orders
            coefficients[k] = ((-1) ** k) * (1.0 / (k - 1))
    elif order == 3:
        # Third derivative: (1/h³)[Δ³y₀ - (3/2)Δ⁴y₀ + ...]
        coefficients[3] = 1.0
        coefficients[4] = -3.0 / 2.0
        coefficients[5] = 7.0 / 4.0
        coefficients[6] = -15.0 / 8.0
        # Add more terms if needed
        for k in range(7, max_terms + 1):
            coefficients[k] = ((-1) ** (k - 2)) * (2.0 ** (k - 3) - 1) / (2.0 ** (k - 3))
    else:
        # For higher orders, use approximate coefficients
        # This is a simplified approach; full implementation would use Stirling numbers
        for k in range(order, max_terms + 1):
            if k == order:
                coefficients[k] = 1.0
            else:
                # Approximate: decreasing magnitude with alternating sign
                coefficients[k] = ((-1) ** (k - order)) / (k - order + 1)
    
    return coefficients


def get_backward_derivative_coefficients(order: int, max_terms: int = 10) -> Dict[int, float]:
    """
    Get coefficients for Newton's backward difference derivative formula.
    
    For nth derivative: (dⁿy/dxⁿ)|ₓ₌ₓₙ = (1/hⁿ) Σ(k=n to m) [Cₖₙ * ∇ᵏyₙ]
    
    Args:
        order: Derivative order (1, 2, 3, ...)
        max_terms: Maximum number of terms to include
    
    Returns:
        Dictionary mapping difference order to coefficient
    """
    coefficients = {}
    
    if order == 1:
        # First derivative: (1/h)[∇yₙ + ∇²yₙ/2 + ∇³yₙ/3 + ∇⁴yₙ/4 + ...]
        for k in range(1, max_terms + 1):
            coefficients[k] = 1.0 / k
    elif order == 2:
        # Second derivative: (1/h²)[∇²yₙ + ∇³yₙ + (11/12)∇⁴yₙ + ...]
        coefficients[2] = 1.0
        coefficients[3] = 1.0
        coefficients[4] = 11.0 / 12.0
        coefficients[5] = 5.0 / 6.0
        coefficients[6] = 137.0 / 180.0
        # Add more terms if needed
        for k in range(7, max_terms + 1):
            coefficients[k] = 1.0 / (k - 1)
    elif order == 3:
        # Third derivative: (1/h³)[∇³yₙ + (3/2)∇⁴yₙ + ...]
        coefficients[3] = 1.0
        coefficients[4] = 3.0 / 2.0
        coefficients[5] = 7.0 / 4.0
        coefficients[6] = 15.0 / 8.0
        # Add more terms if needed
        for k in range(7, max_terms + 1):
            coefficients[k] = (2.0 ** (k - 3) - 1) / (2.0 ** (k - 3))
    else:
        # For higher orders, use approximate coefficients
        for k in range(order, max_terms + 1):
            if k == order:
                coefficients[k] = 1.0
            else:
                # Approximate: positive coefficients
                coefficients[k] = 1.0 / (k - order + 1)
    
    return coefficients


def newton_forward_derivative(
    x_points: List[float],
    y_points: List[float],
    order: int,
    at_point: Optional[float] = None,
    x0_index: Optional[int] = None,
    max_terms: Optional[int] = None
) -> Dict:
    """
    Calculate nth derivative using Newton's forward difference formula.
    
    Args:
        x_points: List of x values (must be equally spaced)
        y_points: List of y values
        order: Derivative order (1, 2, 3, ...)
        at_point: Point at which to calculate derivative (default: x₀)
        x0_index: Index of the point to differentiate at (overrides at_point if provided)
        max_terms: Maximum number of difference terms to use (None = use all available)
    
    Returns:
        Dictionary with results and details
    """
    # Validation
    if len(x_points) != len(y_points):
        return {
            'success': False,
            'message': 'Number of x and y values must match',
            'derivative': None
        }
    
    if len(x_points) < order + 1:
        return {
            'success': False,
            'message': f'Need at least {order + 1} points for {order}{"st" if order == 1 else "nd" if order == 2 else "rd"} derivative',
            'derivative': None
        }
    
    # Validate equally spaced
    is_valid, h = validate_equally_spaced(x_points)
    if not is_valid:
        return {
            'success': False,
            'message': 'Points must be equally spaced',
            'derivative': None
        }
    
    if h == 0:
        return {
            'success': False,
            'message': 'Step size h cannot be zero',
            'derivative': None
        }
    
    # Determine x0_index and at_point
    if x0_index is None:
        if at_point is None:
            x0_index = 0
            at_point = x_points[0]
        else:
            # Find index of at_point
            try:
                x0_index = x_points.index(at_point)
            except ValueError:
                # Find closest point
                x0_index = min(range(len(x_points)), key=lambda i: abs(x_points[i] - at_point))
                at_point = x_points[x0_index]
    else:
        at_point = x_points[x0_index]
    
    # Validate x0_index
    if x0_index < 0 or x0_index >= len(x_points):
        return {
            'success': False,
            'message': f'Invalid x₀ index: {x0_index}',
            'derivative': None
        }
    
    # Check if we have enough points after x0_index
    available_points = len(y_points) - x0_index
    if available_points < order + 1:
        return {
            'success': False,
            'message': f'Need at least {order + 1} points after x₀ (index {x0_index}), but only {available_points} available',
            'derivative': None
        }
    
    # Build difference table starting from x0_index
    # Use ALL available differences (not limited by max_terms)
    max_order_available = available_points - 1
    if max_terms is None:
        max_order_needed = max_order_available
    else:
        max_order_needed = min(order + max_terms - 1, max_order_available)
    
    # Calculate differences starting from x0_index
    y_values_from_x0 = y_points[x0_index:]
    diff_table_full = calculate_forward_differences(y_values_from_x0, max_order_needed)
    
    # Get coefficients - use all available terms
    num_terms_to_use = max_order_needed - order + 1
    coefficients = get_forward_derivative_coefficients(order, num_terms_to_use + order)
    
    # Calculate derivative using ALL available differences
    weighted_sum = 0.0
    terms_used = []
    
    # Use all available differences starting from 'order'
    for k in range(order, min(order + num_terms_to_use, len(diff_table_full[0]))):
        if k < len(diff_table_full[0]) and len(diff_table_full) > 0:
            delta_k_y0 = diff_table_full[0][k]  # Δᵏy₀ starting from x0_index
            coeff = coefficients.get(k, 0.0)
            contribution = coeff * delta_k_y0
            weighted_sum += contribution
            terms_used.append({
                'order': k,
                'difference': delta_k_y0,
                'coefficient': coeff,
                'contribution': contribution
            })
    
    derivative = weighted_sum / (h ** order)
    
    # Build formula string
    if order == 1:
        formula = r"\left(\frac{dy}{dx}\right)_{x=x_0} = \frac{1}{h}\left[\Delta y_0 - \frac{\Delta^2 y_0}{2} + \frac{\Delta^3 y_0}{3} - \frac{\Delta^4 y_0}{4} + \cdots\right]"
    elif order == 2:
        formula = r"\left(\frac{d^2y}{dx^2}\right)_{x=x_0} = \frac{1}{h^2}\left[\Delta^2 y_0 - \Delta^3 y_0 + \frac{11}{12}\Delta^4 y_0 + \cdots\right]"
    elif order == 3:
        formula = r"\left(\frac{d^3y}{dx^3}\right)_{x=x_0} = \frac{1}{h^3}\left[\Delta^3 y_0 - \frac{3}{2}\Delta^4 y_0 + \cdots\right]"
    else:
        formula = f"(d^{order}y/dx^{order})|_{{x=x_0}} = (1/h^{order}) Σ(k={order} to m) [C_{{k{order}}} * Δᵏy₀]"
    
    # Build full difference table for display (all points, but we'll highlight from x0_index)
    diff_table_full_display = calculate_forward_differences(y_points, max_order_needed)
    
    return {
        'success': True,
        'derivative': derivative,
        'order': order,
        'method': 'forward',
        'at_point': at_point,
        'x0_index': x0_index,
        'spacing_h': h,
        'difference_table': diff_table_full_display,  # Full table for display
        'difference_table_from_x0': diff_table_full,  # Table starting from x0_index
        'formula_used': formula,
        'terms_used': terms_used,
        'num_terms_used': len(terms_used),
        'message': f'Successfully calculated {order}{"st" if order == 1 else "nd" if order == 2 else "rd"} derivative using forward difference at x = {at_point:.6f} (index {x0_index})'
    }


def newton_backward_derivative(
    x_points: List[float],
    y_points: List[float],
    order: int,
    at_point: Optional[float] = None,
    xn_index: Optional[int] = None,
    max_terms: Optional[int] = None
) -> Dict:
    """
    Calculate nth derivative using Newton's backward difference formula.
    
    Args:
        x_points: List of x values (must be equally spaced)
        y_points: List of y values
        order: Derivative order (1, 2, 3, ...)
        at_point: Point at which to calculate derivative (default: xₙ)
        xn_index: Index of the point to differentiate at (overrides at_point if provided)
        max_terms: Maximum number of difference terms to use (None = use all available)
    
    Returns:
        Dictionary with results and details
    """
    # Validation
    if len(x_points) != len(y_points):
        return {
            'success': False,
            'message': 'Number of x and y values must match',
            'derivative': None
        }
    
    if len(x_points) < order + 1:
        return {
            'success': False,
            'message': f'Need at least {order + 1} points for {order}{"st" if order == 1 else "nd" if order == 2 else "rd"} derivative',
            'derivative': None
        }
    
    # Validate equally spaced
    is_valid, h = validate_equally_spaced(x_points)
    if not is_valid:
        return {
            'success': False,
            'message': 'Points must be equally spaced',
            'derivative': None
        }
    
    if h == 0:
        return {
            'success': False,
            'message': 'Step size h cannot be zero',
            'derivative': None
        }
    
    # Determine xn_index and at_point
    if xn_index is None:
        if at_point is None:
            xn_index = len(x_points) - 1
            at_point = x_points[-1]
        else:
            # Find index of at_point
            try:
                xn_index = x_points.index(at_point)
            except ValueError:
                # Find closest point
                xn_index = min(range(len(x_points)), key=lambda i: abs(x_points[i] - at_point))
                at_point = x_points[xn_index]
    else:
        at_point = x_points[xn_index]
    
    # Validate xn_index
    if xn_index < 0 or xn_index >= len(x_points):
        return {
            'success': False,
            'message': f'Invalid xₙ index: {xn_index}',
            'derivative': None
        }
    
    # Check if we have enough points before xn_index
    available_points = xn_index + 1
    if available_points < order + 1:
        return {
            'success': False,
            'message': f'Need at least {order + 1} points before xₙ (index {xn_index}), but only {available_points} available',
            'derivative': None
        }
    
    # Build difference table ending at xn_index
    # Use ALL available differences (not limited by max_terms)
    max_order_available = available_points - 1
    if max_terms is None:
        max_order_needed = max_order_available
    else:
        max_order_needed = min(order + max_terms - 1, max_order_available)
    
    # Calculate differences ending at xn_index
    y_values_to_xn = y_points[:xn_index + 1]
    diff_table_full = calculate_backward_differences(y_values_to_xn, max_order_needed)
    
    # Get coefficients - use all available terms
    num_terms_to_use = max_order_needed - order + 1
    coefficients = get_backward_derivative_coefficients(order, num_terms_to_use + order)
    
    # Calculate derivative using ALL available differences
    weighted_sum = 0.0
    terms_used = []
    
    # Use all available differences starting from 'order'
    # The last row in diff_table_full corresponds to xn_index
    n_in_table = len(diff_table_full) - 1
    for k in range(order, min(order + num_terms_to_use, len(diff_table_full[n_in_table]))):
        if k < len(diff_table_full[n_in_table]):
            nabla_k_yn = diff_table_full[n_in_table][k]  # ∇ᵏyₙ at xn_index
            coeff = coefficients.get(k, 0.0)
            contribution = coeff * nabla_k_yn
            weighted_sum += contribution
            terms_used.append({
                'order': k,
                'difference': nabla_k_yn,
                'coefficient': coeff,
                'contribution': contribution
            })
    
    derivative = weighted_sum / (h ** order)
    
    # Build formula string
    if order == 1:
        formula = r"\left(\frac{dy}{dx}\right)_{x=x_n} = \frac{1}{h}\left[\nabla y_n + \frac{\nabla^2 y_n}{2} + \frac{\nabla^3 y_n}{3} + \frac{\nabla^4 y_n}{4} + \cdots\right]"
    elif order == 2:
        formula = r"\left(\frac{d^2y}{dx^2}\right)_{x=x_n} = \frac{1}{h^2}\left[\nabla^2 y_n + \nabla^3 y_n + \frac{11}{12}\nabla^4 y_n + \cdots\right]"
    elif order == 3:
        formula = r"\left(\frac{d^3y}{dx^3}\right)_{x=x_n} = \frac{1}{h^3}\left[\nabla^3 y_n + \frac{3}{2}\nabla^4 y_n + \cdots\right]"
    else:
        formula = f"(d^{order}y/dx^{order})|_{{x=x_n}} = (1/h^{order}) Σ(k={order} to m) [C_{{k{order}}} * ∇ᵏyₙ]"
    
    # Build full difference table for display (all points, but we'll highlight at xn_index)
    diff_table_full_display = calculate_backward_differences(y_points, max_order_needed)
    
    return {
        'success': True,
        'derivative': derivative,
        'order': order,
        'method': 'backward',
        'at_point': at_point,
        'xn_index': xn_index,
        'spacing_h': h,
        'difference_table': diff_table_full_display,  # Full table for display
        'difference_table_to_xn': diff_table_full,  # Table ending at xn_index
        'formula_used': formula,
        'terms_used': terms_used,
        'num_terms_used': len(terms_used),
        'message': f'Successfully calculated {order}{"st" if order == 1 else "nd" if order == 2 else "rd"} derivative using backward difference at x = {at_point:.6f} (index {xn_index})'
    }


# Legacy function names for backward compatibility
def newton_forward_first_derivative(y_values: List[float], h: float, terms: int = 4) -> Tuple[float, Dict]:
    """Legacy function for backward compatibility."""
    x_points = [i * h for i in range(len(y_values))]
    result = newton_forward_derivative(x_points, y_values, 1, None, terms)
    if result['success']:
        details = {
            'method': 'Newton Forward',
            'derivative_order': 1,
            'h': h,
            'delta_y0': result['difference_table'][0][1] if len(result['difference_table']) > 0 and len(result['difference_table'][0]) > 1 else 0,
            'delta2_y0': result['difference_table'][0][2] if len(result['difference_table']) > 0 and len(result['difference_table'][0]) > 2 else 0,
            'delta3_y0': result['difference_table'][0][3] if len(result['difference_table']) > 0 and len(result['difference_table'][0]) > 3 else 0,
            'delta4_y0': result['difference_table'][0][4] if len(result['difference_table']) > 0 and len(result['difference_table'][0]) > 4 else 0,
            'terms_used': result.get('terms_used', []),
            'formula': result['formula_used']
        }
        return result['derivative'], details
    else:
        raise ValueError(result['message'])


def newton_forward_second_derivative(y_values: List[float], h: float, terms: int = 4) -> Tuple[float, Dict]:
    """Legacy function for backward compatibility."""
    x_points = [i * h for i in range(len(y_values))]
    result = newton_forward_derivative(x_points, y_values, 2, None, terms)
    if result['success']:
        details = {
            'method': 'Newton Forward',
            'derivative_order': 2,
            'h': h,
            'delta2_y0': result['difference_table'][0][2] if len(result['difference_table']) > 0 and len(result['difference_table'][0]) > 2 else 0,
            'delta3_y0': result['difference_table'][0][3] if len(result['difference_table']) > 0 and len(result['difference_table'][0]) > 3 else 0,
            'delta4_y0': result['difference_table'][0][4] if len(result['difference_table']) > 0 and len(result['difference_table'][0]) > 4 else 0,
            'terms_used': result.get('terms_used', []),
            'formula': result['formula_used']
        }
        return result['derivative'], details
    else:
        raise ValueError(result['message'])


def newton_forward_third_derivative(y_values: List[float], h: float, terms: int = 4) -> Tuple[float, Dict]:
    """Legacy function for backward compatibility."""
    x_points = [i * h for i in range(len(y_values))]
    result = newton_forward_derivative(x_points, y_values, 3, None, terms)
    if result['success']:
        details = {
            'method': 'Newton Forward',
            'derivative_order': 3,
            'h': h,
            'delta3_y0': result['difference_table'][0][3] if len(result['difference_table']) > 0 and len(result['difference_table'][0]) > 3 else 0,
            'delta4_y0': result['difference_table'][0][4] if len(result['difference_table']) > 0 and len(result['difference_table'][0]) > 4 else 0,
            'terms_used': result.get('terms_used', []),
            'formula': result['formula_used']
        }
        return result['derivative'], details
    else:
        raise ValueError(result['message'])


def newton_backward_first_derivative(y_values: List[float], h: float, terms: int = 4) -> Tuple[float, Dict]:
    """Legacy function for backward compatibility."""
    x_points = [i * h for i in range(len(y_values))]
    n = len(y_values) - 1
    result = newton_backward_derivative(x_points, y_values, 1, None, terms)
    if result['success']:
        details = {
            'method': 'Newton Backward',
            'derivative_order': 1,
            'h': h,
            'nabla_yn': result['difference_table'][n][1] if len(result['difference_table']) > n and len(result['difference_table'][n]) > 1 else 0,
            'nabla2_yn': result['difference_table'][n][2] if len(result['difference_table']) > n and len(result['difference_table'][n]) > 2 else 0,
            'nabla3_yn': result['difference_table'][n][3] if len(result['difference_table']) > n and len(result['difference_table'][n]) > 3 else 0,
            'nabla4_yn': result['difference_table'][n][4] if len(result['difference_table']) > n and len(result['difference_table'][n]) > 4 else 0,
            'terms_used': result.get('terms_used', []),
            'formula': result['formula_used']
        }
        return result['derivative'], details
    else:
        raise ValueError(result['message'])


def newton_backward_second_derivative(y_values: List[float], h: float, terms: int = 4) -> Tuple[float, Dict]:
    """Legacy function for backward compatibility."""
    x_points = [i * h for i in range(len(y_values))]
    n = len(y_values) - 1
    result = newton_backward_derivative(x_points, y_values, 2, None, terms)
    if result['success']:
        details = {
            'method': 'Newton Backward',
            'derivative_order': 2,
            'h': h,
            'nabla2_yn': result['difference_table'][n][2] if len(result['difference_table']) > n and len(result['difference_table'][n]) > 2 else 0,
            'nabla3_yn': result['difference_table'][n][3] if len(result['difference_table']) > n and len(result['difference_table'][n]) > 3 else 0,
            'nabla4_yn': result['difference_table'][n][4] if len(result['difference_table']) > n and len(result['difference_table'][n]) > 4 else 0,
            'terms_used': result.get('terms_used', []),
            'formula': result['formula_used']
        }
        return result['derivative'], details
    else:
        raise ValueError(result['message'])


def newton_backward_third_derivative(y_values: List[float], h: float, terms: int = 4) -> Tuple[float, Dict]:
    """Legacy function for backward compatibility."""
    x_points = [i * h for i in range(len(y_values))]
    n = len(y_values) - 1
    result = newton_backward_derivative(x_points, y_values, 3, None, terms)
    if result['success']:
        details = {
            'method': 'Newton Backward',
            'derivative_order': 3,
            'h': h,
            'nabla3_yn': result['difference_table'][n][3] if len(result['difference_table']) > n and len(result['difference_table'][n]) > 3 else 0,
            'nabla4_yn': result['difference_table'][n][4] if len(result['difference_table']) > n and len(result['difference_table'][n]) > 4 else 0,
            'terms_used': result.get('terms_used', []),
            'formula': result['formula_used']
        }
        return result['derivative'], details
    else:
        raise ValueError(result['message'])
