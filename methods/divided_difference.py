"""
Newton's Divided Difference Interpolation Method
"""

import numpy as np


def divided_difference(x_points, y_points, degree=None):
    """
    Newton's Divided Difference Interpolation
    
    Parameters:
    -----------
    x_points : list or array
        x-coordinates of data points
    y_points : list or array
        y-coordinates of data points
    degree : int, optional
        Polynomial degree (if None, uses all points)
    
    Returns:
    --------
    dict with:
        - success: bool
        - message: str
        - polynomial: function
        - divided_diff_table: 2D list (the divided difference table)
        - coefficients: list (Newton form coefficients)
        - points: list of tuples
    """
    
    x_points = np.array(x_points, dtype=float)
    y_points = np.array(y_points, dtype=float)
    
    n = len(x_points)
    
    # Validate inputs
    if n != len(y_points):
        return {
            'success': False,
            'message': 'Number of x and y points must match',
            'polynomial': None,
            'divided_diff_table': [],
            'coefficients': [],
            'points': []
        }
    
    if n < 2:
        return {
            'success': False,
            'message': 'At least 2 points required',
            'polynomial': None,
            'divided_diff_table': [],
            'coefficients': [],
            'points': []
        }
    
    # Check for duplicate x values
    if len(np.unique(x_points)) != n:
        return {
            'success': False,
            'message': 'Duplicate x values found',
            'polynomial': None,
            'divided_diff_table': [],
            'coefficients': [],
            'points': []
        }
    
    # Limit to specified degree
    if degree is not None:
        if degree >= n:
            degree = n - 1
        n = degree + 1
        x_points = x_points[:n]
        y_points = y_points[:n]
    
    # Build divided difference table
    # Each column represents higher order divided differences
    dd_table = np.zeros((n, n))
    dd_table[:, 0] = y_points  # First column is y values
    
    # Calculate divided differences
    for j in range(1, n):
        for i in range(n - j):
            dd_table[i][j] = (dd_table[i + 1][j - 1] - dd_table[i][j - 1]) / (x_points[i + j] - x_points[i])
    
    # Coefficients are the first row of the table
    coefficients = [dd_table[0][i] for i in range(n)]
    
    # Create polynomial function using Newton's form
    def newton_poly(x):
        """
        Evaluate Newton polynomial at x using nested multiplication (Horner's method adapted)
        P(x) = a0 + a1(x-x0) + a2(x-x0)(x-x1) + ...
        """
        result = coefficients[n - 1]
        for i in range(n - 2, -1, -1):
            result = result * (x - x_points[i]) + coefficients[i]
        return result
    
    # Store points data
    points_data = [{'x': float(x_points[i]), 'y': float(y_points[i])} for i in range(n)]
    
    # Convert table to list for JSON serialization
    table_list = dd_table.tolist()
    
    return {
        'success': True,
        'message': f'Divided difference interpolation successful (degree {n-1})',
        'polynomial': newton_poly,
        'divided_diff_table': table_list,
        'coefficients': coefficients,
        'points': points_data,
        'degree': n - 1,
        'x_points': x_points.tolist()
    }


def format_newton_polynomial(coefficients, x_points):
    """
    Format Newton polynomial in readable form
    P(x) = a0 + a1(x-x0) + a2(x-x0)(x-x1) + ...
    """
    if not coefficients:
        return "P(x) = ?"
    
    terms = []
    terms.append(f"{coefficients[0]:.6f}")
    
    for i in range(1, len(coefficients)):
        # Build the product term (x-x0)(x-x1)...(x-x_{i-1})
        factors = " × ".join([f"(x - {x_points[j]:.4f})" for j in range(i)])
        
        coeff = coefficients[i]
        if coeff >= 0 and i > 0:
            terms.append(f"+ {coeff:.6f} × {factors}")
        else:
            terms.append(f"{coeff:.6f} × {factors}")
    
    return "P(x) = " + " ".join(terms)