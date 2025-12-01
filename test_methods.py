"""
Test suite for all numerical methods
"""

from methods.bisection import bisection
from methods.false_position import false_position
from methods.newton import newton_raphson
from methods.secant import secant
from methods.fixed_point import fixed_point
import numpy as np


def test_all_methods():
    """Test all methods with x^3 - x - 2 = 0"""
    
    print("="*60)
    print("Testing All Root-Finding Methods")
    print("Function: f(x) = x³ - x - 2")
    print("="*60)
    
    # Define function
    f = lambda x: x**3 - x - 2
    f_prime = lambda x: 3*x**2 - 1  # Derivative for Newton
    g = lambda x: (x + 2)**(1/3)     # g(x) for fixed point: x = (x+2)^(1/3)
    
    tol = 1e-6
    
    # Test 1: Bisection
    print("\n1. BISECTION METHOD")
    print("-" * 40)
    result = bisection(f, 1, 2, tol=tol)
    print(f"Success: {result['success']}")
    print(f"Root: {result['root']:.8f}")
    print(f"Iterations: {len(result['iterations'])}")
    print(f"Message: {result['message']}")
    
    # Test 2: False Position
    print("\n2. FALSE POSITION METHOD")
    print("-" * 40)
    result = false_position(f, 1, 2, tol=tol)
    print(f"Success: {result['success']}")
    print(f"Root: {result['root']:.8f}")
    print(f"Iterations: {len(result['iterations'])}")
    print(f"Message: {result['message']}")
    
    # Test 3: Newton-Raphson
    print("\n3. NEWTON-RAPHSON METHOD")
    print("-" * 40)
    result = newton_raphson(f, f_prime, x0=1.5, tol=tol)
    print(f"Success: {result['success']}")
    print(f"Root: {result['root']:.8f}")
    print(f"Iterations: {len(result['iterations'])}")
    print(f"Message: {result['message']}")
    
    # Test 4: Secant
    print("\n4. SECANT METHOD")
    print("-" * 40)
    result = secant(f, x0=1, x1=2, tol=tol)
    print(f"Success: {result['success']}")
    print(f"Root: {result['root']:.8f}")
    print(f"Iterations: {len(result['iterations'])}")
    print(f"Message: {result['message']}")
    
    # Test 5: Fixed Point
    print("\n5. FIXED POINT METHOD")
    print("-" * 40)
    result = fixed_point(g, x0=1.5, tol=tol)
    print(f"Success: {result['success']}")
    print(f"Root: {result['root']:.8f}")
    print(f"Iterations: {len(result['iterations'])}")
    print(f"Message: {result['message']}")
    
    print("\n" + "="*60)
    print("All tests completed! ✅")
    print("="*60)


if __name__ == "__main__":
    test_all_methods()