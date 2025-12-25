import numpy as np

def lagrange_interpolation(x_points, y_points, degree=None):
    """
    Lagrange Interpolation Method
    
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
        - coefficients: list
        - points: list of tuples
        - basis_polynomials: list
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
            'coefficients': [],
            'points': [],
            'basis_polynomials': []
        }
    
    if n < 2:
        return {
            'success': False,
            'message': 'At least 2 points required',
            'polynomial': None,
            'coefficients': [],
            'points': [],
            'basis_polynomials': []
        }
    
    # Check for duplicate x values
    if len(np.unique(x_points)) != n:
        return {
            'success': False,
            'message': 'Duplicate x values found',
            'polynomial': None,
            'coefficients': [],
            'points': [],
            'basis_polynomials': []
        }
    
    # Limit to specified degree
    if degree is not None:
        if degree >= n:
            degree = n - 1
        n = degree + 1
        x_points = x_points[:n]
        y_points = y_points[:n]
    
    def lagrange_basis(i, x):
        """Calculate i-th Lagrange basis polynomial at x"""
        L = 1.0
        for j in range(n):
            if j != i:
                L *= (x - x_points[j]) / (x_points[i] - x_points[j])
        return L
    
    def lagrange_poly(x):
        """Evaluate Lagrange polynomial at x"""
        result = 0.0
        for i in range(n):
            result += y_points[i] * lagrange_basis(i, x)
        return result
    
    # Calculate symbolic representation (coefficients)
    # This expands the Lagrange form to standard polynomial form
    def expand_polynomial():
        """Expand Lagrange polynomial to standard form"""
        # Start with zero polynomial
        coeffs = np.zeros(n)
        
        for i in range(n):
            # For each basis polynomial L_i(x)
            basis_coeffs = np.zeros(n)
            basis_coeffs[0] = 1.0
            
            for j in range(n):
                if j != i:
                    # Multiply by (x - x_j) / (x_i - x_j)
                    new_coeffs = np.zeros(n)
                    factor = 1.0 / (x_points[i] - x_points[j])
                    
                    for k in range(len(basis_coeffs)):
                        if basis_coeffs[k] != 0:
                            # Coefficient of x^(k+1)
                            if k + 1 < n:
                                new_coeffs[k + 1] += basis_coeffs[k] * factor
                            # Coefficient of x^k
                            new_coeffs[k] -= basis_coeffs[k] * x_points[j] * factor
                    
                    basis_coeffs = new_coeffs
            
            # Add weighted basis polynomial to result
            coeffs += y_points[i] * basis_coeffs
        
        return coeffs
    
    try:
        coefficients = expand_polynomial()
    except:
        coefficients = []
    
    # Store basis polynomial information
    basis_info = []
    for i in range(n):
        numerator_terms = []
        denominator = 1.0
        
        for j in range(n):
            if j != i:
                numerator_terms.append(f"(x - {x_points[j]:.10f})")
                denominator *= (x_points[i] - x_points[j])
        
        basis_info.append({
            'index': i,
            'numerator': ' × '.join(numerator_terms),
            'denominator': denominator,
            'coefficient': y_points[i],
            'term': f"{y_points[i]:.10f} × {' × '.join(numerator_terms)} / {denominator:.10f}"
        })
    
    points_data = [{'x': float(x_points[i]), 'y': float(y_points[i])} for i in range(n)]
    
    return {
        'success': True,
        'message': f'Lagrange interpolation successful (degree {n-1})',
        'polynomial': lagrange_poly,
        'coefficients': coefficients.tolist() if len(coefficients) > 0 else [],
        'points': points_data,
        'basis_polynomials': basis_info,
        'degree': n - 1
    }


def format_polynomial(coefficients):
    """Format polynomial coefficients as a readable string"""
    if not coefficients:
        return "P(x) = ?"
    
    terms = []
    degree = len(coefficients) - 1
    
    for i in range(len(coefficients) - 1, -1, -1):
        coeff = coefficients[i]
        if abs(coeff) < 1e-10:
            continue
        
        if i == 0:
            terms.append(f"{coeff:.10f}")
        elif i == 1:
            if coeff > 0 and terms:
                terms.append(f"+ {coeff:.10f}x")
            else:
                terms.append(f"{coeff:.10f}x")
        else:
            if coeff > 0 and terms:
                terms.append(f"+ {coeff:.10f}x^{i}")
            else:
                terms.append(f"{coeff:.10f}x^{i}")
    
    if not terms:
        return "P(x) = 0"
    
    return "P(x) = " + " ".join(terms)