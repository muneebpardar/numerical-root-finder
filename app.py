import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, diff, lambdify, latex, expand, simplify

from methods.bisection import bisection
from methods.false_position import false_position
from methods.newton import newton_raphson
from methods.secant import secant
from methods.fixed_point import fixed_point
from methods.lagrange import lagrange_interpolation, format_polynomial  # NEW IMPORT
from methods.divided_difference import divided_difference, format_newton_polynomial
from methods.jacobi import jacobi_method, check_diagonal_dominance, format_matrix, format_vector, calculate_spectral_radius
from methods.gauss_seidel import gauss_seidel_method, calculate_spectral_radius_gs
from methods.newton_differentiation import (
    newton_forward_derivative, newton_backward_derivative,
    calculate_forward_differences, calculate_backward_differences,
    validate_equally_spaced,
    # Legacy functions for backward compatibility
    newton_forward_first_derivative, newton_forward_second_derivative, newton_forward_third_derivative,
    newton_backward_first_derivative, newton_backward_second_derivative, newton_backward_third_derivative
)
from methods.numerical_integration import numerical_integration
from utils.validators import validate_function, validate_interval, preprocess_function

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_max_precision(value):
    """
    Format a number to maximum precision without unnecessary rounding.
    Uses repr() for floats to show full precision, or converts to string.
    """
    if isinstance(value, (int, float)):
        # Use repr() for maximum precision, but handle very small/large numbers
        if isinstance(value, float):
            # For floats, use repr() which shows maximum precision
            # This will show scientific notation for very small/large numbers
            formatted = repr(value)
            # Remove unnecessary '0' at the end if it's not scientific notation
            if 'e' not in formatted.lower() and formatted.endswith('0'):
                # Keep the decimal point but remove trailing zeros after decimal
                if '.' in formatted:
                    formatted = formatted.rstrip('0').rstrip('.')
            return formatted
        else:
            return str(value)
    elif isinstance(value, complex):
        return repr(value)
    else:
        return str(value)

def format_float_max(value):
    """Format float to maximum precision (15-17 significant digits)"""
    if isinstance(value, float):
        # Use high precision format, but let Python decide the best representation
        # Format with 15 decimal places (Python float precision)
        formatted = f"{value:.15g}"
        # If it's a whole number, show it as integer
        if '.' in formatted and formatted.replace('.', '').replace('-', '').isdigit():
            if float(formatted) == int(float(formatted)):
                return str(int(float(formatted)))
        return formatted
    return str(value)

def show_derivative_info(func_str, x_point=None):
    """
    Helper function to display derivative information for any function.
    Can be called from any method to show differentiation details.
    
    Args:
        func_str: String representation of the function
        x_point: Optional point at which to evaluate (default None)
    """
    try:
        from sympy import symbols, sympify, diff, latex, simplify
        from utils.validators import preprocess_function
        
        x_sym = symbols('x')
        processed = preprocess_function(func_str)
        expr = sympify(processed)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Function:**")
            st.latex(f"f(x) = {latex(expr)}")
        
        with col2:
            f_prime_expr = diff(expr, x_sym)
            simplified = simplify(f_prime_expr)
            st.markdown("**Derivative:**")
            st.latex(f"f'(x) = {latex(simplified)}")
        
        if x_point is not None:
            st.markdown(f"**At x = {x_point:.6f}:**")
            
            f_val = float(expr.subs(x_sym, x_point))
            fprime_val = float(simplified.subs(x_sym, x_point))
            
            col1, col2 = st.columns(2)
            col1.metric("f(x)", f"{f_val:.6f}")
            col2.metric("f'(x)", f"{fprime_val:.6f}")
        
        return True
    except:
        return False


def show_derivative_evaluation_table(f, f_prime, func_str, x_values, title="Derivative Evaluation"):
    """
    Show a table of function and derivative values at specific points.
    
    Args:
        f: Function
        f_prime: Derivative function
        func_str: String representation
        x_values: List of x values to evaluate
        title: Title for the section
    """
    st.markdown(f"#### {title}")
    
    eval_data = []
    for x_val in x_values:
        try:
            f_val = f(x_val)
            fp_val = f_prime(x_val)
            
            # Determine behavior
            if fp_val > 0:
                behavior = "↗️ Increasing"
            elif fp_val < 0:
                behavior = "↘️ Decreasing"
            else:
                behavior = "➡️ Stationary"
            
            eval_data.append({
                'x': f"{x_val:.6f}",
                'f(x)': f"{f_val:.6f}",
                "f'(x)": f"{fp_val:.6f}",
                'Behavior': behavior
            })
        except:
            pass
    
    st.dataframe(pd.DataFrame(eval_data), use_container_width=True, hide_index=True)


st.set_page_config(
    page_title="Numerical Methods Calculator",
    page_icon="🔢",
    layout="wide"
)


# ============================================================================
# REPLACE YOUR EXISTING CSS SECTION IN APP.PY WITH THIS ENHANCED VERSION
# This fixes dark mode visibility issues
# ============================================================================

st.markdown("""
    <style>
    /* ============================================================================
       MODERN UI/UX ENHANCEMENTS - INDUSTRY STANDARD DESIGN
       ============================================================================ */
    
    /* Global Typography & Spacing */
    * {
        font-family: 'Inter', 'Segoe UI', 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* ============================================================================
       HEADER & TITLE ENHANCEMENTS
       ============================================================================ */
    
    .force-header h1 {
        color: white !important;
        -webkit-text-fill-color: white !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
        position: relative;
        z-index: 9999;
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.5px;
    }
    
    h1, h2, h3 {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 700;
        letter-spacing: -0.3px;
    }
    
    h2 {
        font-size: 1.75rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #1a1a1a;
    }
    
    [data-theme="dark"] h2 {
        color: #e0e0e0;
    }
    
    h3 {
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    /* ============================================================================
       FUNCTION INPUT - PREMIUM STYLING
       ============================================================================ */
    
    .stTextArea textarea {
        font-family: 'Fira Code', 'Courier New', 'Consolas', monospace;
        font-size: 16px !important;
        font-weight: 500;
        letter-spacing: 0.3px;
        border: 2px solid #667eea;
        border-radius: 12px;
        padding: 14px 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
    }
    
    .stTextArea textarea:focus {
        border-color: #764ba2;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.25);
        outline: none;
        transform: translateY(-1px);
    }
    
    [data-theme="light"] .stTextArea textarea {
        color: #1a1a1a;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    }
    
    [data-theme="dark"] .stTextArea textarea {
        color: #e0e0e0;
        background: linear-gradient(135deg, #2b2b2b 0%, #1e1e1e 100%);
        border-color: #8b9eea;
    }
    
    /* ============================================================================
       BUTTON ENHANCEMENTS - MODERN & INTERACTIVE
       ============================================================================ */
    
    .stButton > button {
        font-weight: 600;
        border-radius: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.3px;
        border: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        color: inherit !important;
        font-size: 18px !important;
    }
    
    /* Ensure operator buttons are visible */
    .stButton > button[key*="add"],
    .stButton > button[key*="sub"],
    .stButton > button[key*="mul"],
    .stButton > button[key*="div"] {
        font-size: 24px !important;
        font-weight: 700 !important;
        color: #667eea !important;
        min-width: 50px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Primary Button - Premium Gradient */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-size: 16px;
        font-weight: 700;
        padding: 12px 24px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Secondary Buttons */
    .stButton > button:not([kind="primary"]) {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #495057;
        border: 1px solid #dee2e6;
    }
    
    [data-theme="dark"] .stButton > button:not([kind="primary"]) {
        background: linear-gradient(135deg, #2b2b2b 0%, #1e1e1e 100%);
        color: #e0e0e0;
        border: 1px solid #404040;
    }
    
    /* Keypad Buttons */
    .stButton > button[data-testid*="key"] {
        font-size: 14px;
        padding: 10px;
        min-height: 45px;
    }
    
    /* ============================================================================
       METRIC CARDS - PREMIUM DESIGN
       ============================================================================ */
    
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-theme="dark"] [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #8b9eea 0%, #9b6bc2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6c757d;
    }
    
    [data-theme="dark"] [data-testid="stMetricLabel"] {
        color: #adb5bd;
    }
    
    [data-testid="stMetricContainer"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,249,250,0.9) 100%);
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetricContainer"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
    }
    
    [data-theme="dark"] [data-testid="stMetricContainer"] {
        background: linear-gradient(135deg, rgba(43, 43, 43, 0.9) 0%, rgba(30, 30, 30, 0.9) 100%);
        border-color: rgba(139, 158, 234, 0.3);
    }
    
    /* ============================================================================
       TABLE ENHANCEMENTS - MODERN DATA DISPLAY
       ============================================================================ */
    
    .dataframe {
        font-size: 13px;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 14px 12px !important;
        border: none !important;
    }
    
    .dataframe tbody tr {
        transition: all 0.2s ease;
    }
    
    .dataframe tbody tr:hover {
        background-color: rgba(102, 126, 234, 0.05) !important;
        transform: scale(1.01);
    }
    
    .dataframe tbody tr td {
        padding: 12px !important;
        border-bottom: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    [data-theme="dark"] .dataframe {
        background-color: #1e1e1e;
        color: #e0e0e0;
        border-color: rgba(139, 158, 234, 0.2);
    }
    
    [data-theme="dark"] .dataframe tbody tr:hover {
        background-color: rgba(139, 158, 234, 0.1) !important;
    }
    
    /* ============================================================================
       SIDEBAR - PREMIUM DESIGN
       ============================================================================ */
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 2px solid rgba(102, 126, 234, 0.1);
    }
    
    [data-theme="dark"] [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e1e 0%, #2b2b2b 100%);
        border-right-color: rgba(139, 158, 234, 0.2);
    }
    
    [data-testid="stSidebar"] [class*="header"] {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    
    /* Sidebar Radio Buttons */
    .stRadio > div {
        padding: 12px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(248,249,250,0.9) 100%);
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .stRadio > div:hover {
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    [data-theme="dark"] .stRadio > div {
        background: linear-gradient(135deg, rgba(43, 43, 43, 0.9) 0%, rgba(30, 30, 30, 0.9) 100%);
        border-color: rgba(139, 158, 234, 0.3);
    }
    
    /* Sidebar Number Inputs */
    .stNumberInput input {
        border-radius: 10px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .stNumberInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar Text Inputs */
    .stTextInput input {
        border-radius: 10px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* ============================================================================
       ALERT BOXES - MODERN DESIGN
       ============================================================================ */
    
    .stSuccess {
        border-left: 5px solid #28a745;
        padding: 18px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.15);
        margin: 1rem 0;
    }
    
    [data-theme="light"] .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border-left-color: #28a745;
    }
    
    [data-theme="dark"] .stSuccess {
        background: linear-gradient(135deg, #1e4d2b 0%, #2d5a3d 100%);
        color: #a8e6a3;
        border-left-color: #4caf50;
    }
    
    .stError {
        border-left: 5px solid #dc3545;
        padding: 18px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(220, 53, 69, 0.15);
        margin: 1rem 0;
    }
    
    [data-theme="light"] .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border-left-color: #dc3545;
    }
    
    [data-theme="dark"] .stError {
        background: linear-gradient(135deg, #4d1f1f 0%, #5a2525 100%);
        color: #f8a1a1;
        border-left-color: #ff5252;
    }
    
    .stInfo {
        border-left: 5px solid #17a2b8;
        padding: 18px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(23, 162, 184, 0.15);
        margin: 1rem 0;
    }
    
    [data-theme="light"] .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        color: #0c5460;
        border-left-color: #17a2b8;
    }
    
    [data-theme="dark"] .stInfo {
        background: linear-gradient(135deg, #1a3d42 0%, #234a50 100%);
        color: #a8d8e0;
        border-left-color: #26c6da;
    }
    
    .stWarning {
        border-left: 5px solid #ffc107;
        padding: 18px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(255, 193, 7, 0.15);
        margin: 1rem 0;
    }
    
    [data-theme="light"] .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        border-left-color: #ffc107;
    }
    
    [data-theme="dark"] .stWarning {
        background: linear-gradient(135deg, #4d4020 0%, #5a4a25 100%);
        color: #ffe599;
        border-left-color: #ffd54f;
    }
    
    /* ============================================================================
       EXPANDER - MODERN ACCORDION DESIGN
       ============================================================================ */
    
    .streamlit-expanderHeader {
        border-radius: 12px;
        font-weight: 600;
        padding: 14px 16px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
        margin-bottom: 8px;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border-color: rgba(102, 126, 234, 0.4);
        transform: translateX(4px);
    }
    
    [data-theme="light"] .streamlit-expanderHeader {
        color: #1a1a1a;
    }
    
    [data-theme="dark"] .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(43, 43, 43, 0.8) 0%, rgba(30, 30, 30, 0.8) 100%);
        border-color: rgba(139, 158, 234, 0.3);
        color: #e0e0e0;
    }
    
    .streamlit-expanderContent {
        padding: 20px;
        border-radius: 0 0 12px 12px;
        background: rgba(248, 249, 250, 0.5);
        margin-top: -8px;
        border: 1px solid rgba(102, 126, 234, 0.1);
        border-top: none;
    }
    
    [data-theme="dark"] .streamlit-expanderContent {
        background: rgba(30, 30, 30, 0.5);
        border-color: rgba(139, 158, 234, 0.2);
    }
    
    /* ============================================================================
       DOWNLOAD BUTTON - PREMIUM STYLING
       ============================================================================ */
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
        color: white !important;
        font-weight: 600;
        border-radius: 12px;
        padding: 10px 20px;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #20c997 0%, #28a745 100%) !important;
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
        transform: translateY(-2px);
    }
    
    /* ============================================================================
       CODE BLOCKS - ENHANCED STYLING
       ============================================================================ */
    
    code {
        font-family: 'Fira Code', 'Courier New', 'Consolas', monospace;
        font-size: 14px;
        padding: 2px 6px;
        border-radius: 6px;
        background: rgba(102, 126, 234, 0.1);
    }
    
    [data-theme="dark"] code {
        color: #e0e0e0 !important;
        background: rgba(139, 158, 234, 0.15) !important;
    }
    
    pre {
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    [data-theme="dark"] pre {
        background-color: #1e1e1e !important;
        border-color: rgba(139, 158, 234, 0.3);
    }
    
    /* ============================================================================
       LATEX & MATH - ENHANCED DISPLAY
       ============================================================================ */
    
    .katex {
        font-size: 1.1em;
    }
    
    [data-theme="dark"] .katex {
        color: #e0e0e0 !important;
    }
    
    /* ============================================================================
       CHECKBOX & RADIO - MODERN STYLING
       ============================================================================ */
    
    .stCheckbox label {
        font-weight: 500;
        cursor: pointer;
    }
    
    /* ============================================================================
       SECTION DIVIDERS - ELEGANT DESIGN
       ============================================================================ */
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, rgba(102, 126, 234, 0.3) 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* ============================================================================
       SPINNER & LOADING - SMOOTH ANIMATIONS
       ============================================================================ */
    
    .stSpinner > div {
        border-color: #667eea;
    }
    
    /* ============================================================================
       CARD-BASED LAYOUTS
       ============================================================================ */
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,249,250,0.95) 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 1rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    [data-theme="dark"] .metric-card {
        background: linear-gradient(135deg, rgba(43, 43, 43, 0.95) 0%, rgba(30, 30, 30, 0.95) 100%);
        border-color: rgba(139, 158, 234, 0.3);
    }
    
    /* ============================================================================
       SMOOTH SCROLLING
       ============================================================================ */
    
    html {
        scroll-behavior: smooth;
    }
    
    /* ============================================================================
       SELECTBOX & MULTISELECT - ENHANCED
       ============================================================================ */
    
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* ============================================================================
       KEYPAD BUTTONS - GRID LAYOUT ENHANCEMENT
       ============================================================================ */
    
    .stButton > button {
        min-height: 48px;
        font-size: 15px;
    }
    
    /* ============================================================================
       RESPONSIVE DESIGN
       ============================================================================ */
    
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        h1 {
            font-size: 2rem;
        }
        
        h2 {
            font-size: 1.5rem;
        }
    }
    
    /* ============================================================================
       ANIMATIONS
       ============================================================================ */
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stDataFrame {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* ============================================================================
       FOCUS STATES - ACCESSIBILITY
       ============================================================================ */
    
    *:focus-visible {
        outline: 3px solid rgba(102, 126, 234, 0.5);
        outline-offset: 2px;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# ENHANCED HEADER
st.markdown("""
<div class="force-header" style="
    text-align: center;
    padding: 1.5rem 0;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
">
    <h1 style="
        font-size: 2.8rem;
        margin: 0;
        font-weight: 900;
        color: white !important;
        -webkit-text-fill-color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    ">
        🔢 Neeb's Calculator
    </h1>
    <p style="
        color: rgba(255,255,255,0.95);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    ">
        Advanced Root Finding & Interpolation Suite
    </p>
</div>
""", unsafe_allow_html=True)

# INTRO CARD (Overlay Modal) — WITH WORKING BUTTON
if 'intro_dismissed' not in st.session_state:
    st.session_state.intro_dismissed = False

# Check if button was clicked (via query params or session state)
if 'intro_button_clicked' not in st.session_state:
    st.session_state.intro_button_clicked = False

if not st.session_state.intro_dismissed:
    # Inject CSS and HTML for the modal with embedded button
    st.markdown("""
        <style>
        /* Full-page dark overlay */
        .intro-overlay {
            position: fixed;
            top: 0; 
            left: 0;
            width: 100vw; 
            height: 100vh;
            background: rgba(0, 0, 0, 0.85);
            z-index: 999999; 
            display: flex;
            justify-content: center;
            align-items: center;
            animation: fadeIn 0.3s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Intro card container */
        .intro-card {
            width: 500px;
            max-width: 90%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            padding: 2.5rem;
            color: white;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
            border: 3px solid #ffd700;
            animation: slideDown 0.5s ease-out;
        }

        @keyframes slideDown {
            from {
                transform: translateY(-100px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }

        .intro-card h2 {
            color: white !important;
            margin: 0 0 1rem 0;
            font-size: 2.2rem;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }

        .intro-card p {
            color: rgba(255, 255, 255, 0.95) !important;
            font-size: 1.1rem;
            margin: 0 0 1.5rem 0;
            line-height: 1.6;
        }

        .intro-links {
            margin: 1.5rem 0 2rem 0;
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
        }

        .intro-link {
            display: inline-block;
            padding: 12px 28px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 25px;
            color: white !important;
            text-decoration: none;
            border: 2px solid rgba(255, 255, 255, 0.4);
            transition: all 0.3s ease;
            font-weight: 600;
            font-size: 1rem;
        }

        .intro-link:hover {
            background: rgba(255, 255, 255, 0.3);
            border-color: rgba(255, 255, 255, 0.6);
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }

        /* Continue button */
        .intro-continue-btn {
            margin-top: 1.5rem;
            padding: 0.75rem 2rem;
            background: linear-gradient(90deg, #ffd700 0%, #ffed4e 100%);
            color: #764ba2;
            font-size: 1.1rem;
            font-weight: 800;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4);
            transition: all 0.3s ease;
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .intro-continue-btn:hover {
            background: linear-gradient(90deg, #ffed4e 0%, #ffd700 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 215, 0, 0.6);
        }

        .intro-continue-btn:active {
            transform: translateY(0);
        }
        </style>
        
        <div class="intro-overlay">
            <div class="intro-card">
                <h2>👋 Heyyy! I'm Muneeb</h2>
                <p>
                    A post-AI-era coder 🤖 who builds cool stuff and grows CS communities that actually vibe 💥
                </p>
                <div class="intro-links">
                    <a href="https://github.com/muneebpardar" target="_blank" class="intro-link">
                        🔗 GitHub
                    </a>
                    <a href="https://www.linkedin.com/in/muhammad-muneeb-5426a0323" target="_blank" class="intro-link">
                        💼 LinkedIn
                    </a>
                </div>
                <form method="get">
                    <button type="submit" name="continue" value="true" class="intro-continue-btn">
                        🚀 Continue to Calculator
                    </button>
                </form>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Check if the continue button was clicked via query params
    query_params = st.query_params
    if query_params.get("continue") == "true":
        st.session_state.intro_dismissed = True
        # Clear the query param
        st.query_params.clear()
        st.rerun()


    st.stop()
    
# ============================================================================
# MAIN APP CONTENT (only shows after intro is dismissed)
# ============================================================================
st.markdown("---")

    
# Sidebar
# Premium Sidebar Design
st.sidebar.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 1rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
        text-align: center;
    ">
        <h2 style="
            color: white;
            margin: 0;
            font-size: 1.5rem;
            font-weight: 800;
            text-shadow: 0 2px 8px rgba(0,0,0,0.2);
        ">⚙️ Configuration</h2>
    </div>
""", unsafe_allow_html=True)

problem_type = st.sidebar.radio(
    "**📚 Select Problem Type:**",
    ["🎯 Root Finding", "📊 Lagrange Interpolation", "🔢 Divided Difference Interpolation", 
     "🔧 Linear Systems (Jacobi)", "⚡ Linear Systems (Gauss-Seidel)", "📈 Numerical Differentiation", "📐 Numerical Integration"],
    key="problem_type"
)

st.sidebar.markdown("""
    <div style="
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, rgba(102, 126, 234, 0.3) 50%, transparent 100%);
        margin: 1.5rem 0;
    "></div>
""", unsafe_allow_html=True)

# ============================================================================
# ROOT FINDING SECTION
# ============================================================================
if problem_type == "🎯 Root Finding":
    # Method selection
    st.sidebar.markdown("### 🎯 Method Selection")
    method = st.sidebar.selectbox(
        "Choose Method:",
        ["Bisection Method", "False Position Method", "Newton-Raphson Method", "Secant Method", "Fixed Point Method", "🔬 Compare All Methods"]
    )

    # Method info badge
    method_info = {
        "Bisection Method": "🟢 Guaranteed | 🐢 Slow",
        "False Position Method": "🟢 Reliable | 🐇 Faster",
        "Newton-Raphson Method": "🔴 Fast | ⚠️ May Diverge",
        "Secant Method": "🟡 Fast | ⚠️ No Derivative",
        "Fixed Point Method": "🟡 Varies | ⚠️ Needs Good g(x)",
        "🔬 Compare All Methods": "🔬 Benchmark All"
    }

    if method in method_info:
        st.sidebar.caption(method_info[method])

    # Function input
    st.sidebar.markdown("### 🔢 Function Builder")

    # Set default function based on method
    # Check if method changed or if func_input is empty
    if 'last_method' not in st.session_state:
        st.session_state.last_method = method
    
    if st.session_state.last_method != method or 'func_input' not in st.session_state or st.session_state.func_input == "":
        if method == "Fixed Point Method":
            st.session_state.func_input = "(3*x**2 + 3)**(1/4)"
        else:
            st.session_state.func_input = "x**3 - x - 2"
        st.session_state.last_method = method
    
    # Change label based on method
    func_label = "Enter g(x):" if method == "Fixed Point Method" else "Enter f(x):"
    func_help = "Build your iteration function g(x) using the keypad below" if method == "Fixed Point Method" else "Build your function using the keypad below"
    
    func_str = st.sidebar.text_area(
        func_label,
        value=st.session_state.func_input,
        height=100,
        help=func_help,
        key="function_input"
    )

    st.session_state.func_input = func_str

    # Virtual Keypad
    st.sidebar.markdown("#### 🧮 Keypad")

    # Powers & Roots
    col1, col2, col3, col4 = st.sidebar.columns(4)
    if col1.button("x²", use_container_width=True, key="pow2"):
        st.session_state.func_input += "x**2"
        st.rerun()
    if col2.button("x³", use_container_width=True, key="pow3"):
        st.session_state.func_input += "x**3"
        st.rerun()
    if col3.button("xⁿ", use_container_width=True, key="pown"):
        st.session_state.func_input += "x**"
        st.rerun()
    if col4.button("√x", use_container_width=True, key="sqrt"):
        st.session_state.func_input += "sqrt(x)"
        st.rerun()

    # Operators
    col1, col2, col3, col4 = st.sidebar.columns(4)
    if col1.button("+", use_container_width=True, key="add"):
        st.session_state.func_input += " + "
        st.rerun()
    if col2.button("−", use_container_width=True, key="sub"):
        st.session_state.func_input += " - "
        st.rerun()
    if col3.button("×", use_container_width=True, key="mul"):
        st.session_state.func_input += "*"
        st.rerun()
    if col4.button("÷", use_container_width=True, key="div"):
        st.session_state.func_input += "/"
        st.rerun()

    # Trig functions
    col1, col2, col3, col4 = st.sidebar.columns(4)
    if col1.button("sin", use_container_width=True, key="sin"):
        st.session_state.func_input += "sin(x)"
        st.rerun()
    if col2.button("cos", use_container_width=True, key="cos"):
        st.session_state.func_input += "cos(x)"
        st.rerun()
    if col3.button("tan", use_container_width=True, key="tan"):
        st.session_state.func_input += "tan(x)"
        st.rerun()
    if col4.button("π", use_container_width=True, key="pi"):
        st.session_state.func_input += "pi"
        st.rerun()

    # Advanced functions (including inverse trig)
    st.sidebar.markdown("##### Advanced")
    col1, col2, col3, col4 = st.sidebar.columns(4)
    if col1.button("eˣ", use_container_width=True, key="exp"):
        st.session_state.func_input += "exp(x)"
        st.rerun()
    if col2.button("ln", use_container_width=True, key="log"):
        st.session_state.func_input += "log(x)"
        st.rerun()
    if col3.button("|x|", use_container_width=True, key="abs"):
        st.session_state.func_input += "abs(x)"
        st.rerun()
    if col4.button("x", use_container_width=True, key="x"):
        st.session_state.func_input += "x"
        st.rerun()

    # Inverse Trig functions (NEW SECTION)
    st.sidebar.markdown("##### Inverse Trig")
    col1, col2, col3, col4 = st.sidebar.columns(4)
    if col1.button("sin⁻¹", use_container_width=True, key="asin"):
        st.session_state.func_input += "asin(x)"
        st.rerun()
    if col2.button("cos⁻¹", use_container_width=True, key="acos"):
        st.session_state.func_input += "acos(x)"
        st.rerun()
    if col3.button("tan⁻¹", use_container_width=True, key="atan"):
        st.session_state.func_input += "atan(x)"
        st.rerun()
    if col4.button("^", use_container_width=True, key="power"):
        st.session_state.func_input += "^"
        st.rerun()

    # Numbers
    st.sidebar.markdown("##### Numbers")
    col1, col2, col3, col4 = st.sidebar.columns(4)
    if col1.button("7", use_container_width=True, key="7"):
        st.session_state.func_input += "7"
        st.rerun()
    if col2.button("8", use_container_width=True, key="8"):
        st.session_state.func_input += "8"
        st.rerun()
    if col3.button("9", use_container_width=True, key="9"):
        st.session_state.func_input += "9"
        st.rerun()
    if col4.button("(", use_container_width=True, key="lpar"):
        st.session_state.func_input += "("
        st.rerun()

    col1, col2, col3, col4 = st.sidebar.columns(4)
    if col1.button("4", use_container_width=True, key="4"):
        st.session_state.func_input += "4"
        st.rerun()
    if col2.button("5", use_container_width=True, key="5"):
        st.session_state.func_input += "5"
        st.rerun()
    if col3.button("6", use_container_width=True, key="6"):
        st.session_state.func_input += "6"
        st.rerun()
    if col4.button(")", use_container_width=True, key="rpar"):
        st.session_state.func_input += ")"
        st.rerun()

    col1, col2, col3, col4 = st.sidebar.columns(4)
    if col1.button("1", use_container_width=True, key="1"):
        st.session_state.func_input += "1"
        st.rerun()
    if col2.button("2", use_container_width=True, key="2"):
        st.session_state.func_input += "2"
        st.rerun()
    if col3.button("3", use_container_width=True, key="3"):
        st.session_state.func_input += "3"
        st.rerun()
    if col4.button(".", use_container_width=True, key="dot"):
        st.session_state.func_input += "."
        st.rerun()

    # Controls
    st.sidebar.markdown("##### Controls")
    col1, col2, col3, col4 = st.sidebar.columns(4)
    if col1.button("0", use_container_width=True, key="0"):
        st.session_state.func_input += "0"
        st.rerun()
    if col2.button("⌫", use_container_width=True, key="back"):
        st.session_state.func_input = st.session_state.func_input[:-1]
        st.rerun()
    if col3.button("🗑️", use_container_width=True, key="clear"):
        st.session_state.func_input = ""
        st.rerun()
    if col4.button("␣", use_container_width=True, key="space"):
        st.session_state.func_input += " "
        st.rerun()

    st.sidebar.markdown("---")

    # Validate function
    is_valid, f, error_msg = validate_function(func_str)


    # DEBUG OUTPUT
    from utils.validators import preprocess_function
    processed = preprocess_function(func_str)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 DEBUG INFO")
    st.sidebar.code(f"Original:\n{func_str}", language="text")
    st.sidebar.code(f"Processed:\n{processed}", language="text")

    if not is_valid:
        st.sidebar.error(f"❌ Invalid Function")
        st.error(f"⚠️ **Error:** {error_msg}")
        st.error(f"📝 **Processed String:** `{processed}`")
        st.info("💡 **Tip:** Use the keypad buttons or check syntax. Examples: x**2, sin(x), exp(x)")
        st.stop()
    else:
        st.sidebar.success("✅ Valid function")
        
        # Shows preprocessed function
        with st.sidebar.expander("🔍 View Processed Function", expanded=False):
            from utils.validators import preprocess_function
            processed = preprocess_function(func_str)
            st.code(processed, language="python")
            st.caption("This is how your function is interpreted")
        
        # Calculate derivative for Newton method
        try:
            x_sym = symbols('x')
            processed = preprocess_function(func_str)
            expr = sympify(processed)
            f_prime_expr = diff(expr, x_sym)
            f_prime = lambdify(x_sym, f_prime_expr, 'numpy')
        except:
            f_prime = None

    # ====================================================================
    # Method-specific parameters (with interval for Fixed Point)
    # ====================================================================
    st.sidebar.markdown("### 📏 Parameters")

    # Bisection and False Position
    if method in ["Bisection Method", "False Position Method"]:
        col1, col2 = st.sidebar.columns(2)
        a_str = col1.text_input("a (left):", value="-2")
        b_str = col2.text_input("b (right):", value="2")
        
        try:
            a = float(sympify(a_str))
            b = float(sympify(b_str))
        except:
            st.sidebar.error("❌ Invalid interval. Use expressions like 2*pi or pi/4.")
            st.stop()

    # Newton-Raphson
    elif method == "Newton-Raphson Method":
        st.sidebar.markdown("### 📏 Parameters")
        
        # Interval for IVP test (optional)
        st.sidebar.markdown("**Interval for IVP Test (Optional):**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            a_newton = col1.number_input("a (lower bound):", value=None, format="%.10f", key="newton_a", help="Lower bound of interval for sign change check")
        with col2:
            b_newton = col2.number_input("b (upper bound):", value=None, format="%.10f", key="newton_b", help="Upper bound of interval for sign change check")
        
        if a_newton is not None and b_newton is not None:
            if a_newton >= b_newton:
                st.sidebar.error("❌ a must be less than b!")
            else:
                st.sidebar.info(f"💡 IVP test will check for sign change in [{a_newton:.4f}, {b_newton:.4f}]")
        
        st.sidebar.markdown("---")
        
        # Initial guess
        x0 = st.sidebar.number_input("Initial guess (x₀):", value=1.5, format="%.10f")

    # Secant Method
    elif method == "Secant Method":
        col1, col2 = st.sidebar.columns(2)
        x0 = col1.number_input("First guess (x₀):", value=1.0, format="%.10f")
        x1 = col2.number_input("Second guess (x₁):", value=2.0, format="%.10f")

    # Fixed Point
    elif method == "Fixed Point Method":
        # Only show interval and initial guess
        col1, col2 = st.sidebar.columns(2)
        a_str = col1.text_input("a (left endpoint):", value="1")
        b_str = col2.text_input("b (right endpoint):", value="2")
        
        try:
            a = float(sympify(a_str))
            b = float(sympify(b_str))
        except:
            st.sidebar.error("❌ Invalid interval. Use expressions like 0 or 2*pi.")
            st.stop()
        
        # Initial guess
        x0 = st.sidebar.number_input("Initial guess (x₀):", value=1.5, format="%.10f")
        
        st.sidebar.info("💡 Enter **g(x)** directly (the iteration function where x = g(x))")

    # Compare All Methods
    elif method == "🔬 Compare All Methods":
        col1, col2 = st.sidebar.columns(2)
        a_str = col1.text_input("a (for interval methods):", value="-2")
        b_str = col2.text_input("b (for interval methods):", value="2")
        
        try:
            a = float(sympify(a_str))
            b = float(sympify(b_str))
        except:
            st.sidebar.error("❌ Invalid interval. Try expressions like pi/3 or -2*pi.")
            st.stop()
        
        x0_comp = st.sidebar.number_input("Initial guess:", value=1.5, format="%.10f")
        g_str = st.sidebar.text_input("g(x) for Fixed Point:", value="(x + 2)**(1/3)")
        is_valid_g, g, error_msg_g = validate_function(g_str)

    # General parameters for all methods
    tolerance = st.sidebar.number_input("Tolerance:", value=1e-6, format="%.1e", min_value=1e-12)
    max_iter = st.sidebar.number_input("Max Iterations:", value=100, min_value=1, max_value=1000)


    # Display options
    st.sidebar.markdown("### 📊 Display")
    show_graph = st.sidebar.checkbox("Show Graphs", value=True)
    number_format = st.sidebar.radio("Format:", ["Decimal", "Scientific"], horizontal=True)
    

    st.sidebar.markdown("---")
    calculate = st.sidebar.button("🚀 CALCULATE", type="primary", use_container_width=True)

    # Main content
    if calculate:
        st.session_state.has_interacted = True
        
        # COMPARISON MODE
        if method == "🔬 Compare All Methods":
            st.subheader("🔬 Method Comparison")
            
            results = {}
            
            # Run all methods
            with st.spinner("Running all methods..."):
                # Bisection
                valid, msg = validate_interval(f, a, b)
                if valid:
                    results['Bisection'] = bisection(f, a, b, tol=tolerance, max_iter=max_iter)
                
                # False Position
                if valid:
                    results['False Position'] = false_position(f, a, b, tol=tolerance, max_iter=max_iter)
                
                # Newton-Raphson
                if f_prime:
                    results['Newton-Raphson'] = newton_raphson(f, f_prime, x0_comp, tol=tolerance, max_iter=max_iter)
                
                # Secant
                results['Secant'] = secant(f, a, x0_comp, tol=tolerance, max_iter=max_iter)
                
                # Fixed Point
                if is_valid_g:
                    results['Fixed Point'] = fixed_point(g, x0_comp, tol=tolerance, max_iter=max_iter)
            
            # Comparison table
            st.markdown("### 📊 Results Comparison")
            comparison_data = []
            for method_name, result in results.items():
                if result['success']:
                    comparison_data.append({
                        'Method': method_name,
                        'Root': f"{result['root']:.8f}" if number_format == "Decimal" else f"{result['root']:.6e}",
                        'f(root)': f"{f(result['root']):.2e}",
                        'Iterations': len(result['iterations']),
                        'Status': '✅ Success'
                    })
                else:
                    comparison_data.append({
                        'Method': method_name,
                        'Root': 'N/A',
                        'f(root)': 'N/A',
                        'Iterations': len(result['iterations']),
                        'Status': f"❌ {result['message']}"
                    })
            
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            # Find best method
            successful = [r for r in results.items() if r[1]['success']]
            if successful:
                fastest = min(successful, key=lambda x: len(x[1]['iterations']))
                st.success(f"🏆 **Fastest Method:** {fastest[0]} with {len(fastest[1]['iterations'])} iterations")
            
            if show_graph and successful:
                st.markdown("### 📉 Convergence Comparison")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                colors = ['blue', 'green', 'red', 'orange', 'purple']
                for idx, (method_name, result) in enumerate(successful):
                    if result['success']:
                        iters = [it['n'] for it in result['iterations']]
                        
                        # Extract error based on iteration structure
                        if 'error' in result['iterations'][0]:
                            errors = [it['error'] for it in result['iterations']]
                        else:
                            # Calculate error from root approximation
                            final_root = result['root']
                            if 'c' in result['iterations'][0]:
                                errors = [abs(it['c'] - final_root) for it in result['iterations']]
                            elif 'xₙ₊₁' in result['iterations'][0]:
                                errors = [abs(it['xₙ₊₁'] - final_root) for it in result['iterations']]
                            elif 'x₂' in result['iterations'][0]:
                                errors = [abs(it['x₂'] - final_root) for it in result['iterations']]
                            else:
                                continue
                        
                        ax.semilogy(iters, errors, '-o', linewidth=2, markersize=6, 
                                  label=method_name, color=colors[idx % len(colors)])
                
                ax.axhline(y=tolerance, color='black', linestyle='--', linewidth=1.5, label=f'Tolerance = {tolerance:.1e}')
                ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
                ax.set_ylabel('Error (log scale)', fontsize=12, fontweight='bold')
                ax.set_title('Convergence Rate Comparison', fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3, which='both')
                
                st.pyplot(fig)
            
            # Individual method details
            st.markdown("---")
            st.markdown("### 📋 Detailed Results")
            
            for method_name, result in results.items():
                with st.expander(f"{method_name} - Click to expand"):
                    if result['success']:
                        st.success(result['message'])
                        df = pd.DataFrame(result['iterations'])
                        df_display = df.copy()
                        if number_format == "Scientific":
                            for col in df_display.select_dtypes(include=[np.number]).columns:
                                df_display[col] = df_display[col].map(lambda x: f"{x:.6e}")
                        else:
                            for col in df_display.select_dtypes(include=[np.number]).columns:
                                df_display[col] = df_display[col].map(lambda x: f"{x:.10f}")
                        st.dataframe(df_display, use_container_width=True, height=300)
                    else:
                        st.error(result['message'])
        
        # SINGLE METHOD MODE
        else:
            # Run selected method
            if method == "Bisection Method":
                valid, msg = validate_interval(f, a, b)
                if not valid:
                    st.error(f"❌ {msg}")
                    st.stop()
                
                with st.spinner("Calculating..."):
                    result = bisection(f, a, b, tol=tolerance, max_iter=max_iter)
            
            elif method == "False Position Method":
                valid, msg = validate_interval(f, a, b)
                if not valid:
                    st.error(f"❌ {msg}")
                    st.stop()
                
                with st.spinner("Calculating..."):
                    result = false_position(f, a, b, tol=tolerance, max_iter=max_iter)
            
            elif method == "Newton-Raphson Method":
                if f_prime is None:
                    st.error("❌ Could not compute derivative automatically")
                    st.stop()
                
                # IVP Test (Initial Value Problem Test)
                st.markdown("---")
                st.markdown("### 🧪 IVP Test (Initial Value Problem Test)")
                st.markdown("**Checking if Newton's method is suitable for the given initial guess:**")
                
                from methods.newton import ivp_test
                # Pass interval if provided
                a_ivp = a_newton if 'a_newton' in locals() and a_newton is not None else None
                b_ivp = b_newton if 'b_newton' in locals() and b_newton is not None else None
                ivp_result = ivp_test(f, f_prime, x0, tolerance, a=a_ivp, b=b_ivp)
                
                with st.expander("📊 **IVP Test Results**", expanded=True):
                    if ivp_result['pass']:
                        if ivp_result.get('warnings'):
                            st.warning(ivp_result['message'])
                        else:
                            st.success(ivp_result['message'])
                    else:
                        st.error(ivp_result['message'])
                    
                    st.markdown("---")
                    st.markdown("#### Test Details:")
                    
                    details = ivp_result.get('details', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'f_x0' in details:
                            st.markdown(f"**f(x₀):** {format_max_precision(details['f_x0'])}")
                        if 'f_prime_x0' in details:
                            st.markdown(f"**f'(x₀):** {format_max_precision(details['f_prime_x0'])}")
                    
                    with col2:
                        if 'step_size' in details:
                            st.markdown(f"**Initial step size |f(x₀)/f'(x₀)|:** {format_max_precision(details['step_size'])}")
                        if 'x1' in details:
                            st.markdown(f"**x₁ (first iteration):** {format_max_precision(details['x1'])}")
                            if 'x1_distance' in details:
                                st.markdown(f"**|x₁ - x₀|:** {format_max_precision(details['x1_distance'])}")
                    
                    # Show sign change check if interval was provided
                    if 'sign_change_check' in details:
                        sign_check = details['sign_change_check']
                        st.markdown("---")
                        st.markdown("#### 🔍 Sign Change Check (Interval Test):")
                        st.markdown(f"**Interval:** [{format_max_precision(sign_check['a'])}, {format_max_precision(sign_check['b'])}]")
                        st.markdown(f"**f(a):** {format_max_precision(sign_check['f(a)'])}")
                        st.markdown(f"**f(b):** {format_max_precision(sign_check['f(b)'])}")
                        
                        if sign_check['sign_change']:
                            st.success(f"✅ **Sign change detected!** f(a) × f(b) < 0. Root exists in interval [{format_max_precision(sign_check['a'])}, {format_max_precision(sign_check['b'])}] (like bisection method).")
                        else:
                            st.warning(f"⚠️ **No sign change.** f(a) × f(b) ≥ 0. Root may not exist in this interval.")
                    
                    if ivp_result.get('warnings'):
                        st.markdown("---")
                        st.markdown("#### ⚠️ Warnings:")
                        for warning in ivp_result['warnings']:
                            st.warning(f"• {warning}")
                    
                    if not ivp_result['pass']:
                        st.markdown("---")
                        st.markdown("#### ❌ Errors:")
                        errors = ivp_result.get('details', {}).get('errors', [])
                        for error in errors:
                            st.error(f"• {error}")
                        st.warning("⚠️ **Warning:** The method may fail or produce incorrect results. Proceed with caution.")
                
                st.markdown("---")
                
                # Show IVP test status before calculation
                if not ivp_result['pass']:
                    st.warning("⚠️ **IVP Test Failed:** The calculation will proceed, but convergence is not guaranteed.")
                
                with st.spinner("Calculating..."):
                    result = newton_raphson(f, f_prime, x0, tol=tolerance, max_iter=max_iter)
            
            elif method == "Secant Method":
                with st.spinner("Calculating..."):
                    result = secant(f, x0, x1, tol=tolerance, max_iter=max_iter)
            
            elif method == "Fixed Point Method":
                # Use input function directly as g(x) (iteration function)
                st.markdown("---")
                st.markdown("### 🔄 Fixed Point Iteration Function")
                
                with st.expander("🔧 **Iteration Function g(x)**", expanded=True):
                    st.markdown("Your iteration function:")
                    st.code(f"g(x) = {func_str}", language="python")
                    st.info("💡 The function you enter is treated as **g(x)** directly, where **x = g(x)**")
                    
                    try:
                        from sympy import symbols, sympify, simplify, latex, lambdify, diff
                        from utils.validators import preprocess_function
                        
                        x_sym = symbols('x')
                        processed = preprocess_function(func_str)
                        g_expr = sympify(processed)
                        
                        st.markdown("**Mathematical form:**")
                        st.latex(f"g(x) = {latex(g_expr)}")
                        
                        # Compute g'(x)
                        g_prime_expr = diff(g_expr, x_sym)
                        
                        st.markdown("**Derivative:**")
                        st.latex(f"g'(x) = {latex(simplify(g_prime_expr))}")
                        
                        # Create callable functions
                        g = lambdify(x_sym, g_expr, 'numpy')
                        g_prime = lambdify(x_sym, g_prime_expr, 'numpy')
                        
                        # Check convergence ONLY at endpoints (a and b)
                        from methods.fixed_point import check_convergence_condition
                        conv_info = check_convergence_condition(g_prime, a, b)
                        
                        best_k = conv_info.get('k', conv_info['max_derivative'])
                        
                        st.markdown("---")
                        st.markdown("### 📊 Convergence Check")
                        st.markdown(f"**Checking convergence at endpoints only:**")
                        st.markdown(f"- At **x = a = {a}**: |g'({a})| = {abs(float(g_prime(a))):.6f}")
                        st.markdown(f"- At **x = b = {b}**: |g'({b})| = {abs(float(g_prime(b))):.6f}")
                        st.markdown(f"- **k = max(|g'(a)|, |g'(b)|) = {best_k:.6f}**")
                        
                        if conv_info['converges']:
                            st.success(f"✅ **Convergence guaranteed!** k = {best_k:.6f} < 1")
                        else:
                            st.error(f"❌ **May not converge!** k = {best_k:.6f} ≥ 1")
                            st.warning("⚠️ The fixed point method requires |g'(x)| < 1 for convergence. Try a different function or interval.")
                        
                        # Create f(x) = x - g(x) for the iteration table (to show residual)
                        # f(x) = 0 when x = g(x), which is the fixed point condition
                        f_expr = x_sym - g_expr
                        f = lambdify(x_sym, f_expr, 'numpy')
                        
                        # Create f(x) = x - g(x) for the iteration table (to show residual)
                        # f(x) = 0 when x = g(x), which is the fixed point condition
                        f_expr = x_sym - g_expr
                        f = lambdify(x_sym, f_expr, 'numpy')
                        
                        # Store for later use in derivative display
                        st.session_state.fixed_point_g_expr = g_expr
                        st.session_state.fixed_point_g_prime_expr = g_prime_expr
                        st.session_state.fixed_point_best_k = best_k
                        
                    except Exception as e:
                        st.error(f"Error processing function: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        st.stop()
                
                st.markdown("---")
                
                # Convergence Test Plot
                st.markdown("### 📊 Convergence Test: Plotting |g'(x)| over [a, b]")
                
                with st.expander("📈 **Visual Convergence Check**", expanded=True):
                    fig_conv, ax_conv = plt.subplots(figsize=(12, 6))
                    
                    x_plot = np.linspace(a, b, 500)
                    g_prime_vals = []
                    
                    for x_val in x_plot:
                        try:
                            g_prime_vals.append(abs(float(g_prime(x_val))))
                        except:
                            g_prime_vals.append(np.nan)
                    
                    ax_conv.plot(x_plot, g_prime_vals, 'b-', linewidth=2.5, label="|g'(x)|")
                    ax_conv.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Convergence threshold (k=1)')
                    ax_conv.axhline(y=best_k, color='green', linestyle=':', linewidth=2, label=f'Max |g\'(x)| = {best_k:.4f}')
                    ax_conv.axvline(x=x0, color='orange', linestyle='-.', linewidth=2, label=f'Initial guess x₀ = {x0:.4f}')
                    
                    ax_conv.fill_between(x_plot, 0, 1, alpha=0.2, color='green', label='Convergence region')
                    ax_conv.set_xlabel('x', fontsize=13, fontweight='bold')
                    ax_conv.set_ylabel("|g'(x)|", fontsize=13, fontweight='bold')
                    ax_conv.set_title("Convergence Condition: |g'(x)| < 1 (checked only at endpoints a and b)", fontsize=15, fontweight='bold')
                    ax_conv.legend(loc='best')
                    ax_conv.grid(True, alpha=0.3)
                    ax_conv.set_ylim([0, min(2, max([v for v in g_prime_vals if not np.isnan(v)]) * 1.1)])
                    
                    st.pyplot(fig_conv)
                    
                    if best_k < 1:
                        st.success(f"✅ **Convergence guaranteed!** k = {best_k:.6f} < 1 (checked at endpoints only)")
                    else:
                        st.error(f"❌ **May not converge!** k = {best_k:.6f} ≥ 1 (checked at endpoints only)")
                
                st.markdown("---")
                
                # Calculate max iterations using formula
                st.markdown("### 🔢 Maximum Iterations Estimate")
                
                with st.expander("📐 **Formula Calculation**", expanded=True):
                    st.markdown("Using the formula:")
                    st.latex(r"n > \frac{\ln(\text{Error}) - \ln[\max(x_0 - a, b - x_0)]}{\ln(k)}")
                    
                    st.markdown("Where:")
                    st.markdown(f"- **Error** (error tolerance) = {tolerance:.1e}")
                    st.markdown(f"- **x₀** (initial guess / first interval) = {x0}")
                    st.markdown(f"- **a** (first interval endpoint) = {a}")
                    st.markdown(f"- **b** (last interval endpoint) = {b}")
                    st.markdown(f"- **k** (highest value found after convergence test at endpoints) = {best_k:.6f}")
                    
                    max_distance = max(abs(x0 - a), abs(b - x0))
                    st.markdown(f"- **max(|x₀-a|, |b-x₀|)** = max({abs(x0-a):.6f}, {abs(b-x0):.6f}) = {max_distance:.6f}")
                    
                    from methods.fixed_point import calculate_max_iterations_formula
                    estimated_iters = calculate_max_iterations_formula(tolerance, x0, a, b, best_k)
                    
                    if estimated_iters is not None:
                        st.markdown("**Calculation:**")
                        numerator_val = np.log(tolerance) - np.log(max_distance)
                        denominator_val = np.log(best_k)
                        result_val = numerator_val / denominator_val
                        st.code(f"""
                                numerator = ln({tolerance:.1e}) - ln({max_distance:.6f}) = {np.log(tolerance):.6f} - {np.log(max_distance):.6f} = {numerator_val:.6f}
                                denominator = ln({best_k:.6f}) = {denominator_val:.6f}
                                n > {numerator_val:.6f} / {denominator_val:.6f} = {result_val:.6f}
                                Therefore: n ≥ {estimated_iters}
                        """)
                        st.success(f"✅ **Estimated maximum iterations needed: n ≥ {estimated_iters}**")
                    else:
                        st.warning("⚠️ Could not estimate iterations (k ≥ 1 or invalid parameters)")
                
                st.markdown("---")
                
                # Run Fixed Point Method
                with st.spinner("Running Fixed Point iteration..."):
                    from methods.fixed_point import fixed_point
                    result = fixed_point(f, g, x0, tol=tolerance, max_iter=max_iter)
            
            # Display results
            if result['success']:
                st.success(f"✅ {result['message']}")
                
                # Add detailed derivative calculation for Fixed Point Method (similar to Newton)
                if method == "Fixed Point Method":
                    st.markdown("---")
                    st.markdown("### 🧮 Detailed Derivative Calculation for g(x)")
                    
                    with st.expander("📐 **Step-by-Step Differentiation of g(x)**", expanded=True):
                        try:
                            from sympy import symbols, sympify, diff, latex, simplify, expand, Derivative
                            from sympy import sin, cos, tan, exp, log, sqrt, asin, acos, atan, Abs
                            from utils.validators import preprocess_function
                            
                            x_sym = symbols('x')
                            
                            # Get stored expressions
                            best_g_expr = st.session_state.get('fixed_point_g_expr', None)
                            best_g_prime_expr = st.session_state.get('fixed_point_g_prime_expr', None)
                            
                            if best_g_expr is None:
                                st.warning("Could not retrieve g(x) expression for detailed display.")
                            else:
                                st.markdown("#### Your Iteration Function:")
                                st.code(f"g(x) = {best_g_expr}", language="python")
                                st.latex(f"g(x) = {latex(best_g_expr)}")
                                
                                st.markdown("---")
                                
                                # Step 1: Show parsed function
                                st.markdown("#### Step 1: Function in Mathematical Form")
                                st.latex(f"g(x) = {latex(best_g_expr)}")
                                
                                st.markdown("---")
                                
                                # Step 2: Identify differentiation rules
                                st.markdown("#### Step 2: Identify Applicable Differentiation Rules")
                                
                                expr_str = str(best_g_expr)
                                rules_used = []
                                
                                # Detect which rules apply
                                if any(op in expr_str for op in ['**', 'Pow']):
                                    rules_used.append(("Power Rule", r"\frac{d}{dx}[x^n] = n \cdot x^{n-1}"))
                                if 'sin' in expr_str and 'asin' not in expr_str:
                                    rules_used.append(("Sine Rule", r"\frac{d}{dx}[\sin(x)] = \cos(x)"))
                                if 'cos' in expr_str and 'acos' not in expr_str:
                                    rules_used.append(("Cosine Rule", r"\frac{d}{dx}[\cos(x)] = -\sin(x)"))
                                if 'tan' in expr_str and 'atan' not in expr_str:
                                    rules_used.append(("Tangent Rule", r"\frac{d}{dx}[\tan(x)] = \sec^2(x) = 1 + \tan^2(x)"))
                                if 'exp' in expr_str:
                                    rules_used.append(("Exponential Rule", r"\frac{d}{dx}[e^x] = e^x"))
                                if 'log' in expr_str:
                                    rules_used.append(("Natural Log Rule", r"\frac{d}{dx}[\ln(x)] = \frac{1}{x}"))
                                if 'sqrt' in expr_str:
                                    rules_used.append(("Square Root Rule", r"\frac{d}{dx}[\sqrt{x}] = \frac{1}{2\sqrt{x}}"))
                                
                                # Always applicable rules
                                rules_used.append(("Constant Rule", r"\frac{d}{dx}[c] = 0"))
                                rules_used.append(("Sum Rule", r"\frac{d}{dx}[f(x) + g(x)] = f'(x) + g'(x)"))
                                rules_used.append(("Constant Multiple", r"\frac{d}{dx}[c \cdot f(x)] = c \cdot f'(x)"))
                                
                                if '*' in expr_str or len(best_g_expr.args) > 1:
                                    rules_used.append(("Product Rule", r"\frac{d}{dx}[f(x) \cdot g(x)] = f'(x) \cdot g(x) + f(x) \cdot g'(x)"))
                                if '/' in expr_str:
                                    rules_used.append(("Quotient Rule", r"\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x) \cdot g(x) - f(x) \cdot g'(x)}{[g(x)]^2}"))
                                
                                rules_used.append(("Chain Rule", r"\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)"))
                                
                                st.markdown("**Differentiation rules that apply to g(x):**")
                                for rule_name, rule_formula in rules_used:
                                    st.latex(f"\\text{{{rule_name}}}: \\quad {rule_formula}")
                            
                                st.markdown("---")
                                
                                # Step 3: Show the derivative symbolically
                                st.markdown("#### Step 3: Apply Differentiation")
                                
                                st.markdown("**Taking the derivative:**")
                                st.latex(f"\\frac{{d}}{{dx}}\\left[{latex(best_g_expr)}\\right]")
                                
                                # Compute derivative (use stored one if available, otherwise recalculate)
                                if best_g_prime_expr is not None:
                                    g_prime_calc_expr = best_g_prime_expr
                                else:
                                    g_prime_calc_expr = diff(best_g_expr, x_sym)
                                
                                # Show term-by-term if it's a sum
                                from sympy import Add
                                if isinstance(best_g_expr, Add):
                                    st.markdown("**Breaking down term by term (Sum Rule):**")
                                    for i, term in enumerate(best_g_expr.args, 1):
                                        term_derivative = diff(term, x_sym)
                                        st.latex(f"\\text{{Term {i}: }} \\frac{{d}}{{dx}}\\left[{latex(term)}\\right] = {latex(term_derivative)}")
                                    
                                    st.markdown("**Combining all terms:**")
                                
                                st.latex(f"g'(x) = {latex(g_prime_calc_expr)}")
                                
                                st.markdown("---")
                                
                                # Step 4: Simplify
                                st.markdown("#### Step 4: Simplify the Derivative")
                                
                                simplified_g_prime = simplify(g_prime_calc_expr)
                                expanded_g_prime = expand(g_prime_calc_expr)
                                
                                if simplified_g_prime != g_prime_calc_expr:
                                    st.markdown("**Simplified form:**")
                                    st.latex(f"g'(x) = {latex(simplified_g_prime)}")
                                
                                if expanded_g_prime != simplified_g_prime and expanded_g_prime != g_prime_calc_expr:
                                    st.markdown("**Expanded form:**")
                                    st.latex(f"g'(x) = {latex(expanded_g_prime)}")
                                
                                # Show final form
                                st.markdown("**Final derivative:**")
                                st.latex(f"g'(x) = {latex(simplified_g_prime)}")
                                
                                # Code format
                                st.markdown("**Python/Code format:**")
                                st.code(f"g'(x) = {simplified_g_prime}", language="python")
                                
                                st.markdown("---")
                                
                                # Step 5: Numerical verification
                                st.markdown("#### Step 5: Verify with Numerical Approximation")
                                
                                st.markdown("""
                                We can verify our symbolic derivative using the **finite difference formula**:
                                """)
                                st.latex(r"g'(x) \approx \frac{g(x+h) - g(x-h)}{2h} \quad \text{(centered difference)}")
                                
                                test_point = x0
                                h = 1e-7
                                
                                # Numerical derivative using centered difference
                                numerical_derivative = (g(test_point + h) - g(test_point - h)) / (2 * h)
                                
                                # Symbolic derivative value
                                symbolic_derivative = float(simplified_g_prime.subs(x_sym, test_point))
                                
                                verification_data = {
                                    'Method': [
                                        'Symbolic (Exact)',
                                        'Centered Difference',
                                        'Absolute Error'
                                    ],
                                    f'g\'({test_point:.6f})': [
                                        f"{symbolic_derivative:.12f}",
                                        f"{numerical_derivative:.12f}",
                                        f"{abs(symbolic_derivative - numerical_derivative):.2e}"
                                    ]
                                }
                                
                                st.dataframe(pd.DataFrame(verification_data), use_container_width=True, hide_index=True)
                                
                                if abs(symbolic_derivative - numerical_derivative) < 1e-6:
                                    st.success("✅ Symbolic derivative verified! Numerical approximation matches.")
                                else:
                                    st.info(f"ℹ️ Difference: {abs(symbolic_derivative - numerical_derivative):.2e} (acceptable for h={h})")
                                
                                st.markdown("---")
                                
                                # Step 6: Derivative at each Fixed Point iteration
                                st.markdown("#### Step 6: Derivative Values During Iteration")
                                
                                st.markdown("""
                                **How Fixed Point Iteration uses the derivative:**
                                
                                The convergence condition requires **|g'(x)| < 1** for all x in the interval.
                                At each iteration, we compute:
                                """)
                                st.latex(r"x_{n+1} = g(x_n)")
                                
                                st.markdown("The derivative **g'(xₙ)** tells us:")
                                st.markdown("- **Convergence rate:** Smaller |g'(x)| means faster convergence")
                                st.markdown("- **Stability:** |g'(x)| < 1 ensures the method converges")
                                
                                derivative_table = []
                                for iteration in result['iterations'][:min(10, len(result['iterations']))]:
                                    x_val = iteration['xₙ']
                                    g_val = g(x_val)
                                    try:
                                        gprime_val = float(simplified_g_prime.subs(x_sym, x_val))
                                    except:
                                        gprime_val = g_prime(x_val)
                                    
                                    derivative_table.append({
                                        'n': iteration['n'],
                                        'xₙ': f"{x_val:.8f}",
                                        'g(xₙ)': f"{g_val:.8f}",
                                        "|g'(xₙ)|": f"{abs(gprime_val):.6f}",
                                        'xₙ₊₁': f"{iteration['xₙ₊₁']:.8f}",
                                        'Relative Error': f"{iteration['Relative Error']:.6e}"
                                    })
                                
                                st.dataframe(pd.DataFrame(derivative_table), use_container_width=True, hide_index=True)
                                
                                st.info("💡 **Observation:** When |g'(xₙ)| < 1, the method converges. Smaller values mean faster convergence.")
                            
                        except Exception as e:
                            st.error(f"Could not perform symbolic differentiation: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                
                if method == "Newton-Raphson Method":
                    st.markdown("---")
                    st.markdown("### 🧮 Detailed Derivative Calculation")
                    
                    with st.expander("📐 **Step-by-Step Differentiation of f(x)**", expanded=True):
                        st.markdown("#### Your Function:")
                        st.code(f"f(x) = {func_str}", language="python")
                        
                        try:
                            from sympy import symbols, sympify, diff, latex, simplify, expand, Derivative
                            from sympy import sin, cos, tan, exp, log, sqrt, asin, acos, atan, Abs
                            from utils.validators import preprocess_function
                            
                            x_sym = symbols('x')
                            processed = preprocess_function(func_str)
                            expr = sympify(processed)
                            
                            # Step 1: Show parsed function
                            st.markdown("#### Step 1: Function in Mathematical Form")
                            st.latex(f"f(x) = {latex(expr)}")
                            
                            st.markdown("---")
                            
                            # Step 2: Identify differentiation rules
                            st.markdown("#### Step 2: Identify Applicable Differentiation Rules")
                            
                            expr_str = str(expr)
                            rules_used = []
                            
                            # Detect which rules apply
                            if any(op in expr_str for op in ['**', 'Pow']):
                                rules_used.append(("Power Rule", r"\frac{d}{dx}[x^n] = n \cdot x^{n-1}"))
                            if 'sin' in expr_str and 'asin' not in expr_str:
                                rules_used.append(("Sine Rule", r"\frac{d}{dx}[\sin(x)] = \cos(x)"))
                            if 'cos' in expr_str and 'acos' not in expr_str:
                                rules_used.append(("Cosine Rule", r"\frac{d}{dx}[\cos(x)] = -\sin(x)"))
                            if 'tan' in expr_str and 'atan' not in expr_str:
                                rules_used.append(("Tangent Rule", r"\frac{d}{dx}[\tan(x)] = \sec^2(x) = 1 + \tan^2(x)"))
                            if 'exp' in expr_str:
                                rules_used.append(("Exponential Rule", r"\frac{d}{dx}[e^x] = e^x"))
                            if 'log' in expr_str:
                                rules_used.append(("Natural Log Rule", r"\frac{d}{dx}[\ln(x)] = \frac{1}{x}"))
                            if 'asin' in expr_str:
                                rules_used.append(("Arcsine Rule", r"\frac{d}{dx}[\sin^{-1}(x)] = \frac{1}{\sqrt{1-x^2}}"))
                            if 'acos' in expr_str:
                                rules_used.append(("Arccosine Rule", r"\frac{d}{dx}[\cos^{-1}(x)] = \frac{-1}{\sqrt{1-x^2}}"))
                            if 'atan' in expr_str:
                                rules_used.append(("Arctangent Rule", r"\frac{d}{dx}[\tan^{-1}(x)] = \frac{1}{1+x^2}"))
                            if 'sqrt' in expr_str:
                                rules_used.append(("Square Root Rule", r"\frac{d}{dx}[\sqrt{x}] = \frac{1}{2\sqrt{x}}"))
                            if 'Abs' in str(expr):
                                rules_used.append(("Absolute Value Rule", r"\frac{d}{dx}[|x|] = \frac{x}{|x|} \text{ for } x \neq 0"))
                            
                            # Always applicable rules
                            rules_used.append(("Constant Rule", r"\frac{d}{dx}[c] = 0"))
                            rules_used.append(("Sum Rule", r"\frac{d}{dx}[f(x) + g(x)] = f'(x) + g'(x)"))
                            rules_used.append(("Constant Multiple", r"\frac{d}{dx}[c \cdot f(x)] = c \cdot f'(x)"))
                            
                            if '*' in expr_str or len(expr.args) > 1:
                                rules_used.append(("Product Rule", r"\frac{d}{dx}[f(x) \cdot g(x)] = f'(x) \cdot g(x) + f(x) \cdot g'(x)"))
                            if '/' in expr_str:
                                rules_used.append(("Quotient Rule", r"\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x) \cdot g(x) - f(x) \cdot g'(x)}{[g(x)]^2}"))
                            
                            rules_used.append(("Chain Rule", r"\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)"))
                            
                            st.markdown("**Differentiation rules that apply to your function:**")
                            for rule_name, rule_formula in rules_used:
                                st.latex(f"\\text{{{rule_name}}}: \\quad {rule_formula}")
                            
                            st.markdown("---")
                            
                            # Step 3: Show the derivative symbolically
                            st.markdown("#### Step 3: Apply Differentiation")
                            
                            st.markdown("**Taking the derivative:**")
                            st.latex(f"\\frac{{d}}{{dx}}\\left[{latex(expr)}\\right]")
                            
                            # Compute derivative
                            f_prime_expr = diff(expr, x_sym)
                            
                            # Show unevaluated derivative
                            st.markdown("**Applying the rules:**")
                            
                            # Try to show term-by-term differentiation for sums
                            from sympy import Add
                            if isinstance(expr, Add):
                                st.markdown("**Breaking down term by term (Sum Rule):**")
                                for i, term in enumerate(expr.args, 1):
                                    term_derivative = diff(term, x_sym)
                                    st.latex(f"\\text{{Term {i}: }} \\frac{{d}}{{dx}}\\left[{latex(term)}\\right] = {latex(term_derivative)}")
                                
                                st.markdown("**Combining all terms:**")
                            
                            st.latex(f"f'(x) = {latex(f_prime_expr)}")
                            
                            st.markdown("---")
                            
                            # Step 4: Simplify
                            st.markdown("#### Step 4: Simplify the Derivative")
                            
                            simplified_derivative = simplify(f_prime_expr)
                            expanded_derivative = expand(f_prime_expr)
                            
                            if simplified_derivative != f_prime_expr:
                                st.markdown("**Simplified form:**")
                                st.latex(f"f'(x) = {latex(simplified_derivative)}")
                            
                            if expanded_derivative != simplified_derivative and expanded_derivative != f_prime_expr:
                                st.markdown("**Expanded form:**")
                                st.latex(f"f'(x) = {latex(expanded_derivative)}")
                            
                            # Show final form
                            st.markdown("**Final derivative:**")
                            st.latex(f"f'(x) = {latex(simplified_derivative)}")
                            
                            # Code format
                            st.markdown("**Python/Code format:**")
                            st.code(f"f'(x) = {simplified_derivative}", language="python")
                            
                            st.markdown("---")
                            
                            # Step 5: Numerical verification
                            st.markdown("#### Step 5: Verify with Numerical Approximation")
                            
                            st.markdown("""
                            We can verify our symbolic derivative using the **finite difference formula**:
                            """)
                            st.latex(r"f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} \quad \text{(centered difference)}")
                            
                            test_point = x0
                            h = 1e-7
                            
                            # Numerical derivative using centered difference
                            numerical_derivative = (f(test_point + h) - f(test_point - h)) / (2 * h)
                            
                            # Symbolic derivative value
                            symbolic_derivative = float(simplified_derivative.subs(x_sym, test_point))
                            
                            # Forward difference for comparison
                            forward_diff = (f(test_point + h) - f(test_point)) / h
                            
                            verification_data = {
                                'Method': [
                                    'Symbolic (Exact)',
                                    'Centered Difference',
                                    'Forward Difference',
                                    'Absolute Error (Centered)',
                                    'Absolute Error (Forward)'
                                ],
                                f'f\'({test_point:.6f})': [
                                    f"{symbolic_derivative:.12f}",
                                    f"{numerical_derivative:.12f}",
                                    f"{forward_diff:.12f}",
                                    f"{abs(symbolic_derivative - numerical_derivative):.2e}",
                                    f"{abs(symbolic_derivative - forward_diff):.2e}"
                                ]
                            }
                            
                            st.dataframe(pd.DataFrame(verification_data), use_container_width=True, hide_index=True)
                            
                            if abs(symbolic_derivative - numerical_derivative) < 1e-6:
                                st.success("✅ Symbolic derivative verified! Numerical approximation matches.")
                            else:
                                st.info(f"ℹ️ Difference: {abs(symbolic_derivative - numerical_derivative):.2e} (acceptable for h={h})")
                            
                            st.markdown("---")
                            
                            # Step 6: Derivative at each Newton iteration
                            st.markdown("#### Step 6: Derivative Values During Iteration")
                            
                            st.markdown("""
                            **How Newton-Raphson uses the derivative:**
                            
                            At each iteration, we compute:
                            """)
                            st.latex(r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}")
                            
                            st.markdown("The derivative **f'(xₙ)** tells us:")
                            st.markdown("- **Direction:** Which way to move (left or right)")
                            st.markdown("- **Step size:** How far to move (inversely proportional to slope)")
                            
                            derivative_table = []
                            for iteration in result['iterations'][:min(10, len(result['iterations']))]:
                                x_val = iteration['xₙ']
                                f_val = iteration['f(xₙ)']
                                fprime_val = iteration["f'(xₙ)"]
                                
                                # Calculate step size
                                step_size = -f_val / fprime_val if fprime_val != 0 else float('inf')
                                
                                derivative_table.append({
                                    'n': iteration['n'],
                                    'xₙ': f"{x_val:.8f}",
                                    'f(xₙ)': f"{f_val:.6e}",
                                    "f'(xₙ)": f"{fprime_val:.6f}",
                                    'Step = -f/f\'': f"{step_size:.6f}",
                                    'xₙ₊₁': f"{iteration['xₙ₊₁']:.8f}"
                                })
                            
                            st.dataframe(pd.DataFrame(derivative_table), use_container_width=True, hide_index=True)
                            
                            st.info("💡 **Observation:** When |f'(xₙ)| is large (steep slope), the step size is small. When |f'(xₙ)| is small (flat region), the step size is large.")
                            
                        except Exception as e:
                            st.error(f"Could not perform symbolic differentiation: {e}")
                            st.info("The method still works using numerical derivative approximation.")
                    
                    # Geometric interpretation
                    with st.expander("📊 **Geometric Interpretation: Tangent Lines**", expanded=False):
                        st.markdown("""
                        #### How Newton-Raphson Works Geometrically:
                        
                        1. **Start at point** (xₙ, f(xₙ))
                        2. **Draw tangent line** with slope f'(xₙ)
                        3. **Find where tangent crosses x-axis** → this is xₙ₊₁
                        4. **Repeat** until we're close enough to the actual root
                        
                        **Tangent Line Equation:**
                        """)
                        st.latex(r"y - f(x_n) = f'(x_n)(x - x_n)")
                        
                        st.markdown("**Where tangent crosses x-axis (y = 0):**")
                        st.latex(r"0 - f(x_n) = f'(x_n)(x - x_n)")
                        st.latex(r"x = x_n - \frac{f(x_n)}{f'(x_n)} = x_{n+1}")
                        
                        st.markdown("---")
                        
                        # Show first 3 tangent lines
                        st.markdown("**First 3 Tangent Lines:**")
                        
                        for i, iteration in enumerate(result['iterations'][:3], 1):
                            x_n = iteration['xₙ']
                            f_n = iteration['f(xₙ)']
                            fp_n = iteration["f'(xₙ)"]
                            x_next = iteration['xₙ₊₁']
                            
                            st.markdown(f"**Iteration {i}:**")
                            st.latex(f"\\text{{Point: }} ({x_n:.6f}, {f_n:.6f})")
                            st.latex(f"\\text{{Tangent: }} y - ({f_n:.6f}) = ({fp_n:.6f})(x - {x_n:.6f})")
                            st.latex(f"\\text{{Crosses x-axis at: }} x = {x_next:.6f}")
                            st.markdown("")
                    
                    # Why derivative matters
                    with st.expander("🎯 **Why the Derivative is Critical**", expanded=False):
                        st.markdown("""
                        #### When Newton-Raphson Works Best:
                        
                        ✅ **Good conditions:**
                        - f'(x) is **non-zero** near the root
                        - f'(x) is **continuous**
                        - Initial guess x₀ is **close to root**
                        - Function is **smooth** (differentiable)
                        
                        ❌ **Problems occur when:**
                        - **f'(x) = 0** (horizontal tangent) → division by zero
                        - **f'(x) ≈ 0** (nearly flat) → huge steps, possible divergence
                        - **f'(x) changes rapidly** → unpredictable behavior
                        - **Multiple roots nearby** → may jump between them
                        
                        #### Example Issues:
                        """)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Problem: Zero Derivative**")
                            st.code("f(x) = x²\nf'(x) = 2x\nAt x=0: f'(0) = 0 ❌", language="python")
                            st.caption("Newton-Raphson fails at x=0")
                        
                        with col2:
                            st.markdown("**Problem: Flat Region**")
                            st.code("f(x) = x³ - 2x + 2\nf'(x) = 3x² - 2\nNear x=0.8: f'≈0.08 ⚠️", language="python")
                            st.caption("Very small derivative → large steps")


                elif method == "Secant Method":
                    st.markdown("---")
                    st.markdown("### 🔢 Finite Difference Approximation (Derivative-Free)")
                    
                    with st.expander("📐 **How Secant Approximates the Derivative**", expanded=True):
                        st.markdown("""
                        The Secant method **doesn't need** the actual derivative formula! Instead, it **approximates** 
                        the derivative using two function values.
                        
                        #### Finite Difference Formula:
                        """)
                        
                        st.latex(r"f'(x_n) \approx \frac{f(x_n) - f(x_{n-1})}{x_n - x_{n-1}}")
                        
                        st.markdown("""
                        This is called the **backward difference** approximation.
                        
                        #### Comparison with Newton-Raphson:
                        """)
                        
                        comparison = pd.DataFrame({
                            'Aspect': [
                                'Derivative Needed?',
                                'Number of Points',
                                'Formula',
                                'Convergence Rate',
                                'Function Evaluations/Iter',
                                'Best Use Case'
                            ],
                            'Newton-Raphson': [
                                'Yes (analytical)',
                                '1 point',
                                'xₙ₊₁ = xₙ - f(xₙ)/f\'(xₙ)',
                                'Quadratic (~2.0)',
                                '2 (f and f\')',
                                'When derivative is easy'
                            ],
                            'Secant': [
                                'No (numerical)',
                                '2 points',
                                'xₙ₊₁ = xₙ - f(xₙ)·Δx/Δf',
                                'Superlinear (~1.618)',
                                '1 (only f)',
                                'When derivative is hard'
                            ]
                        })
                        
                        st.dataframe(comparison, use_container_width=True, hide_index=True)
                        
                        st.markdown("---")
                        
                        # Show derivative approximations at each step
                        st.markdown("#### Derivative Approximations During Iteration")
                        
                        approx_table = []
                        for iteration in result['iterations'][:min(10, len(result['iterations']))]:
                            x0_val = iteration['x₀']
                            x1_val = iteration['x₁']
                            f0_val = iteration['f(x₀)']
                            f1_val = iteration['f(x₁)']
                            
                            # Calculate approximate derivative
                            if abs(x1_val - x0_val) > 1e-14:
                                approx_derivative = (f1_val - f0_val) / (x1_val - x0_val)
                            else:
                                approx_derivative = float('inf')
                            
                            # Calculate actual derivative if possible
                            try:
                                from sympy import symbols, sympify, diff, lambdify
                                from utils.validators import preprocess_function
                                x_sym = symbols('x')
                                processed = preprocess_function(func_str)
                                expr = sympify(processed)
                                f_prime_expr = diff(expr, x_sym)
                                f_prime_func = lambdify(x_sym, f_prime_expr, 'numpy')
                                actual_derivative = f_prime_func(x1_val)
                                error = abs(approx_derivative - actual_derivative)
                            except:
                                actual_derivative = None
                                error = None
                            
                            row_data = {
                                'n': iteration['n'],
                                'x₀': f"{x0_val:.6f}",
                                'x₁': f"{x1_val:.6f}",
                                'Δx': f"{x1_val - x0_val:.6e}",
                                'Δf': f"{f1_val - f0_val:.6e}",
                                "f'≈Δf/Δx": f"{approx_derivative:.6f}"
                            }
                            
                            if actual_derivative is not None:
                                row_data["f'(actual)"] = f"{actual_derivative:.6f}"
                                row_data['Error'] = f"{error:.6e}"
                            
                            approx_table.append(row_data)
                        
                        st.dataframe(pd.DataFrame(approx_table), use_container_width=True, hide_index=True)
                        
                        st.info("💡 **Key Insight:** As iterations progress and xₙ₊₁ gets closer to xₙ, the finite difference approximation becomes more accurate!")
                    
                    with st.expander("🎓 **Mathematical Derivation**", expanded=False):
                        st.markdown("""
                        #### How Secant Method is Derived:
                        
                        **Step 1:** Start with Newton-Raphson formula
                        """)
                        st.latex(r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}")
                        
                        st.markdown("**Step 2:** Approximate f'(xₙ) using two points")
                        st.latex(r"f'(x_n) \approx \frac{f(x_n) - f(x_{n-1})}{x_n - x_{n-1}}")
                        
                        st.markdown("**Step 3:** Substitute into Newton formula")
                        st.latex(r"x_{n+1} = x_n - \frac{f(x_n)}{\frac{f(x_n) - f(x_{n-1})}{x_n - x_{n-1}}}")
                        
                        st.markdown("**Step 4:** Simplify")
                        st.latex(r"x_{n+1} = x_n - f(x_n) \cdot \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}")
                        
                        st.markdown("This is the **Secant Method formula**! ✨")
                        
                        st.markdown("---")
                        st.markdown("#### Geometric Interpretation:")
                        st.markdown("""
                        - **Newton:** Uses tangent line (requires derivative)
                        - **Secant:** Uses secant line through two points (no derivative needed)
                        """)
                        
                        st.latex(r"\text{Secant line slope} = \frac{f(x_n) - f(x_{n-1})}{x_n - x_{n-1}}")

                # ============================================================================
                # COMPARISON MODE - Derivative comparison
                # ============================================================================

                if method == "🔬 Compare All Methods" and calculate:
                    st.markdown("---")
                    st.markdown("### 🔬 Derivative Requirements Comparison")
                    
                    with st.expander("📊 **How Each Method Handles Derivatives**", expanded=True):
                        derivative_comparison = {
                            'Method': [
                                'Bisection',
                                'False Position',
                                'Newton-Raphson',
                                'Secant',
                                'Fixed Point'
                            ],
                            'Needs f\'(x)?': [
                                '❌ No',
                                '❌ No',
                                '✅ Yes (analytical)',
                                '❌ No',
                                '⚠️ Indirectly (for convergence check)'
                            ],
                            'How it works': [
                                'Uses sign changes only',
                                'Uses linear interpolation',
                                'Uses tangent line (f\'(x))',
                                'Approximates f\'(x) numerically',
                                'Uses iteration function g(x)'
                            ],
                            'Derivative Formula': [
                                'Not needed',
                                'Not needed',
                                'Must compute f\'(x) symbolically',
                                'f\'(x) ≈ (f(xₙ)-f(xₙ₋₁))/(xₙ-xₙ₋₁)',
                                'Need g\'(x) < 1 for convergence'
                            ],
                            'Best When': [
                                'Derivative unknown',
                                'Derivative unknown',
                                'Derivative easy to compute',
                                'Derivative hard/expensive',
                                'Can transform to x = g(x)'
                            ]
                        }
                        
                        st.dataframe(pd.DataFrame(derivative_comparison), use_container_width=True, hide_index=True)
                        
                        st.markdown("---")
                        st.markdown("#### Derivative Analysis for Current Function:")
                        st.code(f"f(x) = {func_str}", language="python")
                        
                        try:
                            from sympy import symbols, sympify, diff, latex, simplify
                            from utils.validators import preprocess_function
                            
                            x_sym = symbols('x')
                            processed = preprocess_function(func_str)
                            expr = sympify(processed)
                            f_prime_expr = diff(expr, x_sym)
                            simplified_derivative = simplify(f_prime_expr)
                            
                            st.markdown("**Computed derivative:**")
                            st.latex(f"f'(x) = {latex(simplified_derivative)}")
                            
                            # Evaluate at the root found by the fastest method
                            successful = [r for r in results.items() if r[1]['success']]
                            if successful:
                                fastest = min(successful, key=lambda x: len(x[1]['iterations']))
                                root_val = fastest[1]['root']
                                
                                fprime_at_root = float(simplified_derivative.subs(x_sym, root_val))
                                
                                st.markdown(f"**At the root (x ≈ {root_val:.6f}):**")
                                st.latex(f"f'({root_val:.6f}) = {fprime_at_root:.6f}")
                                
                                # Analysis
                                if abs(fprime_at_root) > 10:
                                    st.info("💡 Large |f'(x)| at root → Steep function → Good for Newton-Raphson (small steps)")
                                elif abs(fprime_at_root) < 0.1:
                                    st.warning("⚠️ Small |f'(x)| at root → Flat region → Newton-Raphson may take large steps")
                                else:
                                    st.success("✅ Moderate |f'(x)| at root → Good for all derivative-based methods")
                        
                        except Exception as e:
                            st.info("Could not compute symbolic derivative for comparison.")

                # Metrics
                col1, col2, col3 = st.columns(3)
                root_val = result['root']
                
                col1.metric("Root", f"{root_val:.8f}" if number_format == "Decimal" else f"{root_val:.6e}")
                
                if method == "Fixed Point Method":
                    col2.metric("f(root)", f"{f(root_val):.2e}")
                    # Add verification
                    st.markdown("---")
                    verification_error = abs(root_val - g(root_val))
                    if verification_error < tolerance * 10:
                        st.success(f"✅ Fixed point verified: |x - g(x)| = {verification_error:.2e}")
                    else:
                        st.warning(f"⚠️ Fixed point error: |x - g(x)| = {verification_error:.2e}")
                else:
                    col2.metric("f(root)", f"{f(root_val):.2e}")
                
                col3.metric("Iterations", len(result['iterations']))
                
                st.markdown("---")
                
                # Iteration table
                st.subheader("📊 Iteration Table")
                df = pd.DataFrame(result['iterations'])
                
                # Ensure all numeric columns are real (not complex)
                for col in df.select_dtypes(include=[np.number]).columns:
                    df[col] = df[col].apply(lambda x: np.real(x) if isinstance(x, complex) or np.iscomplexobj(x) else float(x))
                
                # Custom formatting for Newton-Raphson Method
                if method == "Newton-Raphson Method":
                    # Reorder columns: n, xₙ, f(xₙ), f'(xₙ), xₙ₊₁, error (error at the end)
                    df_renamed = df.copy()
                    
                    # Remove step column if it exists
                    if 'step' in df_renamed.columns:
                        df_renamed = df_renamed.drop(columns=['step'])
                    
                    # Reorder columns: n, xₙ, f(xₙ), f'(xₙ), xₙ₊₁, error
                    # Define desired order (error will be moved to end separately)
                    column_order = ['n', 'xₙ', 'f(xₙ)', "f'(xₙ)", 'xₙ₊₁']
                    # Only include columns that exist
                    column_order = [col for col in column_order if col in df_renamed.columns]
                    # Get remaining columns (excluding error, which will be added at the end)
                    remaining_cols = [col for col in df_renamed.columns if col not in column_order and col != 'error']
                    # Add error at the end if it exists
                    if 'error' in df_renamed.columns:
                        final_order = column_order + remaining_cols + ['error']
                    else:
                        final_order = column_order + remaining_cols
                    
                    # Reorder the dataframe
                    df_renamed = df_renamed[final_order]
                    
                    df = df_renamed
                
                # Create a display copy without altering original data
                df_display = df.copy()

                if number_format == "Scientific":
                    for col in df_display.select_dtypes(include=[np.number]).columns:
                        df_display[col] = df_display[col].map(lambda x: f"{float(x):.6e}")
                else:
                    for col in df_display.select_dtypes(include=[np.number]).columns:
                        df_display[col] = df_display[col].map(lambda x: f"{float(x):.10f}")

                st.dataframe(df_display, use_container_width=True, height=400)
                
                # Show formula explanations for Newton-Raphson
                if method == "Newton-Raphson Method":
                    st.markdown("---")
                    with st.expander("📐 **Column Formulas**", expanded=False):
                        st.markdown("""
                        **Column Definitions:**
                        - **xₙ**: Current approximation at iteration n
                        - **f(xₙ)**: Function value at xₙ
                        - **f'(xₙ)**: Derivative value at xₙ
                        - **xₙ₊₁**: Next approximation = xₙ - f(xₙ)/f'(xₙ)
                        - **error**: Relative error = |(xₙ₊₁ - xₙ) / xₙ₊₁|
                        """)
                        st.latex(r"\text{Next approximation: } x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}")
                        st.latex(r"\text{Error formula: } \left|\frac{x_{n+1} - x_n}{x_{n+1}}\right|")
                
                # Download
                csv = pd.DataFrame(result['iterations']).to_csv(index=False)
                st.download_button("📥 Download CSV", csv, f"{method.replace(' ', '_')}_results.csv", "text/csv")
                
                # Derivative Behavior Analysis for Newton-Raphson (after iteration table)
                if method == "Newton-Raphson Method" and result['success']:
                    st.markdown("---")
                    st.markdown("### 📈 Derivative Behavior Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Derivative Magnitudes")
                        
                        # Extract derivative values
                        iterations_list = [it['n'] for it in result['iterations']]
                        derivative_vals = [abs(it["f'(xₙ)"]) for it in result['iterations']]
                        
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.plot(iterations_list, derivative_vals, 'b-o', linewidth=2, markersize=8)
                        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
                        ax.set_ylabel("|f'(x)|", fontsize=12, fontweight='bold')
                        ax.set_title("Derivative Magnitude During Convergence", fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        ax.set_yscale('log')
                        
                        st.pyplot(fig)
                        
                        st.caption("📊 Shows how steep the function is at each iteration point")
                    
                    with col2:
                        st.markdown("#### Step Size Analysis")
                        
                        step_sizes = [abs(it['xₙ₊₁'] - it['xₙ']) for it in result['iterations']]
                        
                        fig2, ax2 = plt.subplots(figsize=(8, 5))
                        ax2.plot(iterations_list, step_sizes, 'r-s', linewidth=2, markersize=8)
                        ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
                        ax2.set_ylabel('Step Size |xₙ₊₁ - xₙ|', fontsize=12, fontweight='bold')
                        ax2.set_title("Step Size During Convergence", fontsize=14, fontweight='bold')
                        ax2.grid(True, alpha=0.3)
                        ax2.set_yscale('log')
                        
                        st.pyplot(fig2)
                        
                        st.caption("📊 Shows how step size decreases as we approach the root")
                    
                    st.markdown("#### 🔍 Observations:")
                    
                    avg_derivative = np.mean([abs(it["f'(xₙ)"]) for it in result['iterations']])
                    max_step = max(step_sizes)
                    min_step = min([s for s in step_sizes if s > 0])
                    
                    insights = []
                    
                    if avg_derivative > 5:
                        insights.append("✅ **Steep function** (avg |f'| = {:.2f}) → Small, controlled steps".format(avg_derivative))
                    elif avg_derivative < 0.5:
                        insights.append("⚠️ **Flat function** (avg |f'| = {:.2f}) → Large steps, use caution".format(avg_derivative))
                    else:
                        insights.append("✅ **Moderate slope** (avg |f'| = {:.2f}) → Well-behaved convergence".format(avg_derivative))
                    
                    if max_step / min_step > 100:
                        insights.append("📊 **Large step variation** (ratio = {:.1f}) → Function changes character significantly".format(max_step/min_step))
                    else:
                        insights.append("📊 **Consistent steps** → Function has uniform behavior near root")
                    
                    for insight in insights:
                        st.markdown(insight)
                
                # Graph
                if show_graph:
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📈 Function Plot")
                        fig, ax = plt.subplots(figsize=(7, 5))
                        
                        # Determine plot range
                        if 'c' in result['iterations'][0]:
                            points = [it['c'] for it in result['iterations']]
                        elif 'xₙ₊₁' in result['iterations'][0]:
                            points = [it['xₙ₊₁'] for it in result['iterations']]
                        elif 'x₂' in result['iterations'][0]:
                            points = [it['x₂'] for it in result['iterations']]
                        else:
                            points = [root_val]
                        
                        x_min, x_max = min(points) - 1, max(points) + 1
                        x_plot = np.linspace(x_min, x_max, 1000)
                        
                        try:
                            y_plot = f(x_plot)
                            ax.plot(x_plot, y_plot, 'b-', linewidth=2.5)
                            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                            ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
                            ax.plot(root_val, f(root_val), 'r*', markersize=20, label=f'Root = {root_val:.6f}')
                            ax.axvline(root_val, color='red', linestyle=':', alpha=0.5)
                            
                            ax.set_xlabel('x', fontsize=12, fontweight='bold')
                            ax.set_ylabel('f(x)', fontsize=12, fontweight='bold')
                            ax.legend()
                            ax.grid(alpha=0.3)
                            
                            st.pyplot(fig)
                        except:
                            st.error("Could not plot function")
    else:
        # Welcome screen when no calculation has been run
        st.info("👈 **Getting Started:** Build your function using the keypad in the sidebar, set parameters, and click 'Calculate Root'")

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("### 🎓 About Bisection Method")
            st.markdown("""
            The **Bisection Method** is a root-finding algorithm that repeatedly bisects an interval 
            and selects a subinterval where a root must lie.
            
            #### Algorithm Steps:
            1. Start with interval **[a, b]** where **f(a) · f(b) < 0**
            2. Calculate midpoint: **c = (a + b) / 2**
            3. Evaluate **f(c)**
            4. Check convergence: If **|f(c)| < ε** → Root found
            5. Update interval:
               - If **f(a) · f(c) < 0** → Root in **[a, c]**, set **b = c**
               - Else → Root in **[c, b]**, set **a = c**
            6. Repeat until convergence
            
            #### Properties:
            - ✅ **Always converges** if initial interval is valid
            - ✅ **Simple and robust**
            - ⚠️ **Linear convergence** (slow for high precision)
            - ⚠️ Requires **continuous function**
            - ⚠️ Requires **sign change** in interval
            """)

        with col2:
            st.markdown("### 📝 Quick Examples")
            st.code("x**3 - x - 2", language="python")
            st.caption("Cubic equation")

            st.code("x**2 - 4", language="python")
            st.caption("Root at x = 2")

            st.code("cos(x) - x", language="python")
            st.caption("Transcendental")

            st.code("exp(x) - 3", language="python")
            st.caption("Exponential")

            st.code("sin(x) - 0.5", language="python")
            st.caption("Trigonometric")

            st.markdown("---")
            st.markdown("### 🎯 Key Formulas")
            st.latex(r"\text{Error}_n = \frac{b-a}{2^n}")
            st.caption("Error after n iterations")

            st.latex(r"n_{max} = \left\lceil \frac{\ln(b-a) - \ln(\epsilon)}{\ln(2)} \right\rceil")
            st.caption("Maximum iterations needed")
# ============================================================================
# LAGRANGE INTERPOLATION SECTION
# ============================================================================
elif problem_type == "📊 Lagrange Interpolation":
    st.sidebar.markdown("### 📊 Lagrange Interpolation")
    
    # Degree selection
    degree = st.sidebar.selectbox(
        "Polynomial Degree:",
        [1, 2, 3],
        index=2,
        help="Select the degree of interpolating polynomial"
    )
    
    st.sidebar.info(f"💡 You need {degree + 1} data points for degree {degree} polynomial")
    
    # Number of points
    num_points = degree + 1
    
    st.sidebar.markdown(f"### 📍 Enter {num_points} Data Points")
    
    # Initialize session state for points
    if 'x_points' not in st.session_state:
        st.session_state.x_points = [0.0] * 4
    if 'y_points' not in st.session_state:
        st.session_state.y_points = [0.0] * 4
    
    # Default examples based on degree (Lagrange)
    default_points = {
        1: {   # degree 1 → uses first 2 points
            'x': [8.1, 8.3],
            'y': [16.94410, 17.56492]
        },
        2: {   # degree 2 → first 3 points
            'x': [8.1, 8.3, 8.6],
            'y': [16.94410, 17.56492, 18.50515]
        },
        3: {   # degree 3 → full example (4 points)
            'x': [8.1, 8.3, 8.6, 8.7],
            'y': [16.94410, 17.56492, 18.50515, 18.82091]
        }
    }

    
    # Quick example button
    if st.sidebar.button("📝 Load Example", use_container_width=True):
        st.session_state.x_points = default_points[degree]['x'] + [0.0] * (4 - len(default_points[degree]['x']))
        st.session_state.y_points = default_points[degree]['y'] + [0.0] * (4 - len(default_points[degree]['y']))
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Input points
    x_points = []
    y_points = []
    
    for i in range(num_points):
        st.sidebar.markdown(f"**Point {i + 1}:**")
        col1, col2 = st.sidebar.columns(2)
        
        x_val = col1.number_input(
            f"x{i}:",
            value=float(st.session_state.x_points[i]) if i < len(st.session_state.x_points) else 0.0,
            format="%.10f",
            key=f"x_{i}"
        )
        y_val = col2.number_input(
            f"y{i}:",
            value=float(st.session_state.y_points[i]) if i < len(st.session_state.y_points) else 0.0,
            format="%.10f",
            key=f"y_{i}"
        )
        
        x_points.append(x_val)
        y_points.append(y_val)
    
    # Update session state
    st.session_state.x_points = x_points + [0.0] * (4 - len(x_points))
    st.session_state.y_points = y_points + [0.0] * (4 - len(y_points))
    
    st.sidebar.markdown("---")
    
    # Evaluation point
    st.sidebar.markdown("### 🎯 Evaluation")
    eval_x = st.sidebar.number_input(
        "Evaluate P(x) at:",
        value=0.5,
        format="%.10f",
        help="Enter x value to evaluate the polynomial"
    )
    
    # Display options
    st.sidebar.markdown("### 📊 Display")
    show_details = st.sidebar.checkbox("Show Basis Polynomials", value=True)
    show_graph = st.sidebar.checkbox("Show Graph", value=True)
    show_algebra = st.sidebar.checkbox("Show Detailed Algebra", value=True, 
                                       help="Show step-by-step algebraic expansion")
    
    st.sidebar.markdown("---")
    calculate_interp = st.sidebar.button("🚀 INTERPOLATE", type="primary", use_container_width=True)
    
    # Main content for Lagrange
    if calculate_interp:
        st.session_state.has_interacted = True
        # Validate unique x values
        if len(set(x_points)) != len(x_points):
            st.error("❌ Error: All x values must be unique!")
            st.stop()
        
        # Compute interpolation
        with st.spinner("Calculating interpolation..."):
            result = lagrange_interpolation(x_points, y_points, degree=degree)
        
        if result['success']:
            st.success(f"✅ {result['message']}")
            
            # Display polynomial
            st.markdown("### 📐 Interpolating Polynomial")
            
            # Show in expanded form
            try:
                from sympy import symbols, expand, simplify, latex
                x = symbols('x')
                full_expr = 0
                
                for i in range(len(x_points)):
                    yi = y_points[i]
                    num_sym = 1
                    denom = 1
                    
                    for j, xj in enumerate(x_points):
                        if i != j:
                            num_sym *= (x - xj)
                            denom *= (x_points[i] - xj)
                    
                    full_expr += yi * (num_sym / denom)
                
                expanded_expr = expand(full_expr)
                simplified_expr = simplify(expanded_expr)
                
                st.latex(f"P(x) = {latex(simplified_expr)}")
                
                # Also show in code format
                st.code(str(simplified_expr), language="python")
                
            except Exception as e:
                st.warning(f"Symbolic form unavailable. Showing numerical form.")
                if result['coefficients']:
                    poly_str = format_polynomial(result['coefficients'])
                    st.code(poly_str, language="python")
            
            st.markdown("---")
            
            # DETAILED STEP-BY-STEP SOLUTION
            st.markdown("### 📝 Detailed Step-by-Step Solution")
            
            # Step 1: Show the formula
            with st.expander("📖 **Step 1: Understanding Lagrange Formula**", expanded=True):
                st.markdown("""
                The Lagrange interpolating polynomial is given by:
                """)
                st.latex(r"P(x) = \sum_{i=0}^{n} y_i \cdot L_i(x)")
                st.markdown("""
                Where each basis polynomial is:
                """)
                st.latex(r"L_i(x) = \prod_{\substack{j=0\\j \neq i}}^{n} \frac{x - x_j}{x_i - x_j}")
                st.markdown(f"""
                For our problem:
                - We have **{len(x_points)} data points**
                - Polynomial degree: **{degree}**
                - We need to compute **{len(x_points)} basis polynomials**
                """)
            
            # Step 2: Calculate each basis polynomial
            with st.expander("🔧 **Step 2: Calculating Lagrange Basis Polynomials L_i(x)**", expanded=True):
                st.markdown("We calculate each basis polynomial step by step:")
                
                for i in range(len(x_points)):
                    st.markdown(f"#### Basis Polynomial L_{i}(x)")
                    
                    # Show which point this basis is for
                    st.info(f"For point ({x_points[i]}, {y_points[i]}): This polynomial equals 1 at x={x_points[i]} and 0 at all other data points.")
                    
                    # Build numerator
                    numerator_terms = []
                    denominator_terms = []
                    numerator_latex = []
                    denominator_vals = []
                    
                    for j in range(len(x_points)):
                        if i != j:
                            numerator_terms.append(f"(x - {x_points[j]})")
                            numerator_latex.append(f"(x - {x_points[j]})")
                            
                            denom_val = x_points[i] - x_points[j]
                            denominator_terms.append(f"({x_points[i]} - {x_points[j]})")
                            denominator_vals.append(denom_val)
                    
                    # Show the formula for this basis
                    numerator_str = r" \cdot ".join(numerator_latex)
                    denominator_str = r" \cdot ".join([f"({x_points[i]} - {x_points[j]})" for j in range(len(x_points)) if i != j])
                    
                    st.latex(f"L_{i}(x) = \\frac{{{numerator_str}}}{{{denominator_str}}}")
                    
                    # Calculate denominator
                    denominator_product = 1
                    denom_calculation = " × ".join([f"({x_points[i]} - {x_points[j]})" for j in range(len(x_points)) if i != j])
                    denom_result = " × ".join([f"{x_points[i] - x_points[j]:.4f}" for j in range(len(x_points)) if i != j])
                    
                    for j in range(len(x_points)):
                        if i != j:
                            denominator_product *= (x_points[i] - x_points[j])
                    
                    st.markdown(f"""
                    **Denominator calculation:**
```
                    {denom_calculation} = {denom_result} = {denominator_product:.6f}
```
                    """)
                    
                    # Final basis polynomial
                    numerator_display = " × ".join(numerator_terms)
                    st.success(f"**Result:** L_{i}(x) = [{numerator_display}] / {denominator_product:.6f}")
                    
                    st.markdown("---")
            
            # Step 3: Multiply by y-values
            with st.expander("✖️ **Step 3: Multiply Each Basis by Corresponding y-value**", expanded=True):
                st.markdown("Now we multiply each basis polynomial by its corresponding y-value:")
                
                terms_for_sum = []
                
                for i in range(len(x_points)):
                    yi = y_points[i]
                    
                    numerator_terms = []
                    denominator = 1
                    
                    for j in range(len(x_points)):
                        if i != j:
                            numerator_terms.append(f"(x - {x_points[j]})")
                            denominator *= (x_points[i] - x_points[j])
                    
                    numerator_str = " × ".join(numerator_terms)
                    
                    st.markdown(f"""
                    **Term {i+1}:** y_{i} × L_{i}(x)
```
                    = {yi:.6f} × [{numerator_str}] / {denominator:.6f}
```
                    """)
                    
                    # Store for final sum
                    terms_for_sum.append(f"({yi:.10f} × [{numerator_str}] / {denominator:.10f})")
            
            # Step 4: Sum all terms
            with st.expander("➕ **Step 4: Sum All Terms to Get P(x)**", expanded=True):
                st.markdown("The final polynomial is the sum of all weighted basis polynomials:")
                
                st.latex("P(x) = " + " + ".join([f"y_{i} \\cdot L_{i}(x)" for i in range(len(x_points))]))
                
                st.markdown("**Substituting the values:**")
                
                # Show full expansion
                full_sum = "\n     + ".join(terms_for_sum)
                st.code(f"P(x) = {full_sum}", language="text")
            
            # Step 5: Symbolic expansion with detailed algebra
            if show_algebra:
                with st.expander("📐 **Step 5: Algebraic Expansion and Simplification**", expanded=True):
                    st.markdown("Now let's expand the polynomial algebraically, step by step:")
                
                    try:
                        from sympy import symbols, expand, simplify, latex, collect, Poly
                        x = symbols('x')
                        
                        # Show the sum we're starting with
                        st.markdown("#### 5.1 Starting Expression")
                        st.markdown("We have:")
                        
                        sum_parts = []
                        for i in range(len(x_points)):
                            yi = y_points[i]
                            num_parts = []
                            denom = 1
                            
                            for j in range(len(x_points)):
                                if i != j:
                                    num_parts.append(f"(x - {x_points[j]})")
                                    denom *= (x_points[i] - x_points[j])
                            
                            numerator_str = " × ".join(num_parts)
                            sum_parts.append(f"({yi:.6f} × [{numerator_str}] / {denom:.6f})")
                        
                        full_expression = "\n     + ".join(sum_parts)
                        st.code(f"P(x) = {full_expression}", language="text")
                        
                        st.markdown("---")
                        
                        # Step 5.2: Expand each term individually
                        st.markdown("#### 5.2 Expand Each Term Separately")
                        
                        expanded_terms = []
                        expanded_terms_symbolic = []
                        
                        for i in range(len(x_points)):
                            st.markdown(f"**Term {i+1}:**")
                            
                            yi = y_points[i]
                            num_sym = 1
                            denom = 1
                            
                            # Build the numerator symbolically
                            for j, xj in enumerate(x_points):
                                if i != j:
                                    num_sym *= (x - xj)
                                    denom *= (x_points[i] - xj)
                            
                            # Show the fraction
                            num_parts = []
                            for j in range(len(x_points)):
                                if i != j:
                                    num_parts.append(f"(x - {x_points[j]})")
                            
                            numerator_str = " × ".join(num_parts)
                            st.code(f"Term {i+1} = {yi:.6f} × [{numerator_str}] / {denom:.6f}")
                            
                            # Expand the numerator
                            expanded_num = expand(num_sym)
                            st.markdown(f"Expanding the numerator **[{numerator_str}]**:")
                            st.latex(f"= {latex(expanded_num)}")
                            
                            # Divide by denominator
                            term_result = yi * (num_sym / denom)
                            expanded_term = expand(term_result)
                            
                            st.markdown(f"Now divide by {denom:.6f} and multiply by {yi:.6f}:")
                            st.latex(f"= {latex(expanded_term)}")
                            
                            # Store for later
                            expanded_terms_symbolic.append(expanded_term)
                            
                            # Show decimal form
                            from sympy import N
                            decimal_form = N(expanded_term, 6)
                            st.code(f"≈ {decimal_form}", language="python")
                            
                            expanded_terms.append(str(decimal_form))
                            
                            st.markdown("---")
                        
                        # Step 5.3: Combine all terms
                        st.markdown("#### 5.3 Combine All Expanded Terms")
                        st.markdown("Now we add all the expanded terms together:")
                        
                        combination_latex = " + ".join([f"({latex(term)})" for term in expanded_terms_symbolic])
                        st.latex(f"P(x) = {combination_latex}")
                        
                        # Full expansion
                        full_expr = sum(expanded_terms_symbolic)
                        expanded_expr = expand(full_expr)
                        
                        st.markdown("**After expansion:**")
                        st.latex(f"P(x) = {latex(expanded_expr)}")
                        
                        st.markdown("---")
                        
                        # Step 5.4: Collect like terms
                        st.markdown("#### 5.4 Collect Like Terms")
                        st.markdown("Grouping terms by powers of x:")
                        
                        collected_expr = collect(expanded_expr, x)
                        
                        # Extract coefficients
                        try:
                            poly_obj = Poly(collected_expr, x)
                            coeffs = poly_obj.all_coeffs()
                            
                            # Show term by term
                            degree_of_poly = len(coeffs) - 1
                            term_breakdown = []
                            
                            for power in range(degree_of_poly, -1, -1):
                                coeff = float(coeffs[degree_of_poly - power])
                                
                                if abs(coeff) > 1e-10:
                                    if power == 0:
                                        term_breakdown.append({
                                            'Power': 'x⁰ (constant)',
                                            'Coefficient': f"{coeff:.6f}",
                                            'Term': f"{coeff:.6f}"
                                        })
                                    elif power == 1:
                                        term_breakdown.append({
                                            'Power': 'x¹',
                                            'Coefficient': f"{coeff:.6f}",
                                            'Term': f"{coeff:.6f}x"
                                        })
                                    else:
                                        term_breakdown.append({
                                            'Power': f'x^{power}',
                                            'Coefficient': f"{coeff:.6f}",
                                            'Term': f"{coeff:.6f}x^{power}"
                                        })
                            
                            st.table(pd.DataFrame(term_breakdown))
                            
                        except Exception as e:
                            st.warning(f"Could not extract coefficients: {e}")
                        
                        st.markdown("---")
                        
                        # Step 5.5: Final simplified form
                        st.markdown("#### 5.5 Final Simplified Polynomial")
                        
                        simplified_expr = simplify(collected_expr)
                        
                        # Show in mathematical notation
                        st.latex(f"P(x) = {latex(simplified_expr)}")
                        
                        # Show in code format
                        st.code(f"P(x) = {simplified_expr}", language="python")
                        
                        # Show in standard polynomial format
                        try:
                            poly_obj = Poly(simplified_expr, x)
                            coeffs = poly_obj.all_coeffs()
                            
                            # Build readable polynomial string
                            poly_terms = []
                            degree_of_poly = len(coeffs) - 1
                            
                            for power in range(degree_of_poly, -1, -1):
                                coeff = float(coeffs[degree_of_poly - power])
                                
                                if abs(coeff) < 1e-10:
                                    continue
                                
                                if power == 0:
                                    if coeff > 0 and poly_terms:
                                        poly_terms.append(f"+ {coeff:.10f}")
                                    else:
                                        poly_terms.append(f"{coeff:.10f}")
                                elif power == 1:
                                    if coeff > 0 and poly_terms:
                                        poly_terms.append(f"+ {coeff:.10f}x")
                                    elif coeff < 0:
                                        poly_terms.append(f"- {abs(coeff):.10f}x")
                                    else:
                                        poly_terms.append(f"{coeff:.10f}x")
                                else:
                                    if coeff > 0 and poly_terms:
                                        poly_terms.append(f"+ {coeff:.10f}x^{power}")
                                    elif coeff < 0:
                                        poly_terms.append(f"- {abs(coeff):.10f}x^{power}")
                                    else:
                                        poly_terms.append(f"{coeff:.10f}x^{power}")
                            
                            readable_poly = " ".join(poly_terms)
                            
                            st.success(f"**Final Answer:** P(x) = {readable_poly}")
                            
                        except:
                            pass
                        
                        st.markdown("---")
                        
                        # Step 5.6: Numerical verification of expansion
                        st.markdown("#### 5.6 Verify the Expansion")
                        st.markdown("Let's verify our expansion is correct by testing at a random point:")
                        
                        test_x = (x_points[0] + x_points[-1]) / 2  # Midpoint
                        
                        # Evaluate using original Lagrange form
                        original_val = result['polynomial'](test_x)
                        
                        # Evaluate using expanded form
                        expanded_val = float(simplified_expr.subs(x, test_x))
                        
                        verification_data = {
                            'Method': ['Lagrange Form', 'Expanded Form', 'Difference'],
                            'Value at x=' + f'{test_x:.4f}': [
                                f"{original_val:.8f}",
                                f"{expanded_val:.8f}",
                                f"{abs(original_val - expanded_val):.2e}"
                            ]
                        }
                        
                        st.dataframe(pd.DataFrame(verification_data), use_container_width=True, hide_index=True)
                        
                        if abs(original_val - expanded_val) < 1e-10:
                            st.success("✅ Perfect match! The expansion is correct.")
                        else:
                            st.warning(f"⚠️ Small numerical difference: {abs(original_val - expanded_val):.2e}")
                        
                    except Exception as e:
                        st.error(f"Error in symbolic expansion: {e}")
                        st.info("Showing numerical approximation instead:")
                        
                        if result['coefficients']:
                            poly_str = format_polynomial(result['coefficients'])
                            st.code(poly_str, language="python")
                
                # Step 6: Verification
                with st.expander("✅ **Step 6: Verify the Polynomial Passes Through All Points**", expanded=True):
                    st.markdown("Let's verify that P(x) actually passes through all our data points:")
                    
                    verification_steps = []
                    for i, (xi, yi) in enumerate(zip(x_points, y_points)):
                        p_xi = result['polynomial'](xi)
                        error = abs(p_xi - yi)
                        
                        verification_steps.append({
                            'Point': f"({xi}, {yi})",
                            'P(x) at x=' + f"{xi}": f"{p_xi:.8f}",
                            'Expected y': f"{yi:.8f}",
                            'Error': f"{error:.2e}",
                            'Status': '✅ Pass' if error < 1e-6 else '❌ Fail'
                        })
                    
                    st.dataframe(pd.DataFrame(verification_steps), use_container_width=True, hide_index=True)
                    
                    max_error = max([abs(result['polynomial'](xi) - yi) for xi, yi in zip(x_points, y_points)])
                    
                    if max_error < 1e-10:
                        st.success(f"✅ **Perfect!** Maximum error: {max_error:.2e} (essentially zero)")
                    elif max_error < 1e-6:
                        st.success(f"✅ **Excellent!** Maximum error: {max_error:.2e}")
                    else:
                        st.warning(f"⚠️ Maximum error: {max_error:.2e}")
                
                st.markdown("---")
                
                # Quick summary table
                st.markdown("### 📊 Quick Reference")
                
                summary_steps = []
                for i in range(len(x_points)):
                    yi = y_points[i]
                    num_parts = []
                    denom = 1
                    
                    for j, xj in enumerate(x_points):
                        if i != j:
                            num_parts.append(f"(x - {xj})")
                            denom *= (x_points[i] - xj)
                    
                    numerator_str = " × ".join(num_parts) if num_parts else "1"
                    
                    summary_steps.append({
                        'i': i,
                        'Point': f"({x_points[i]}, {yi})",
                        'Basis L_i(x)': f"[{numerator_str}] / {denom:.4f}",
                        'Term': f"{yi:.4f} × L_{i}(x)"
                    })
                
                st.dataframe(pd.DataFrame(summary_steps), use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # Evaluate at given point
                eval_result = result['polynomial'](eval_x)
                
                # Evaluation step-by-step
                st.markdown(f"### 🎯 Evaluating P({eval_x})")
                
                with st.expander(f"📊 **Detailed Calculation at x = {eval_x}**", expanded=False):
                    st.markdown(f"Let's evaluate the polynomial at x = {eval_x}:")
                    
                    total = 0
                    calc_steps = []
                    
                    for i in range(len(x_points)):
                        # Calculate L_i(eval_x)
                        L_i_val = 1.0
                        for j in range(len(x_points)):
                            if i != j:
                                L_i_val *= (eval_x - x_points[j]) / (x_points[i] - x_points[j])
                        
                        term_val = y_points[i] * L_i_val
                        total += term_val
                        
                        calc_steps.append({
                            'Step': f"Term {i+1}",
                            'Formula': f"y_{i} × L_{i}({eval_x})",
                            'Calculation': f"{y_points[i]:.4f} × {L_i_val:.6f}",
                            'Result': f"{term_val:.10f}"
                        })
                    
                    st.dataframe(pd.DataFrame(calc_steps), use_container_width=True, hide_index=True)
                    
                    st.markdown(f"""
                    **Final Sum:**
    ```
                    P({eval_x}) = {' + '.join([f"{step['Result']}" for step in calc_steps])}
                            = {total:.6f}
    ```
                    """)
                    # Step 5: Symbolic expansion with detailed algebra
            else:
                # Simple version
                with st.expander("📐 **Step 5: Final Polynomial**", expanded=True):
                    st.markdown("Expanded and simplified polynomial:")
                    
                    try:
                        from sympy import symbols, expand, simplify, latex
                        x = symbols('x')
                        full_expr = 0
                        
                        for i in range(len(x_points)):
                            yi = y_points[i]
                            num_sym = 1
                            denom = 1
                            
                            for j, xj in enumerate(x_points):
                                if i != j:
                                    num_sym *= (x - xj)
                                    denom *= (x_points[i] - xj)
                            
                            full_expr += yi * (num_sym / denom)
                        
                        simplified_expr = simplify(expand(full_expr))
                        
                        st.latex(f"P(x) = {latex(simplified_expr)}")
                        st.code(str(simplified_expr), language="python")
                        
                    except Exception as e:
                        st.warning("Symbolic form unavailable")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Polynomial Degree", result['degree'])
            col2.metric(f"P({eval_x})", f"{eval_result:.6f}")
            col3.metric("Data Points Used", len(result['points']))
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Polynomial Degree", result['degree'])
            col2.metric(f"P({eval_x})", f"{eval_result:.6f}")
            col3.metric("Data Points Used", len(result['points']))
            
            st.markdown("---")
            
            # Data points table
            st.markdown("### 📍 Input Data Points")
            points_df = pd.DataFrame(result['points'])
            st.dataframe(points_df, use_container_width=True, hide_index=True)
            
            # Detailed basis polynomials
            if show_details:
                st.markdown("---")
                st.markdown("### 🔧 Lagrange Basis Polynomials (Detailed)")
                
                for basis in result['basis_polynomials']:
                    with st.expander(f"L_{basis['index']}(x) - Basis Polynomial {basis['index']}"):
                        st.markdown(f"**Numerator:** {basis['numerator']}")
                        st.markdown(f"**Denominator:** {basis['denominator']:.6f}")
                        st.markdown(f"**Coefficient (y_{basis['index']}):** {basis['coefficient']:.6f}")
                        st.markdown("---")
                        st.code(f"L_{basis['index']}(x) = {basis['term']}")
            
            # Graph
            if show_graph:
                st.markdown("---")
                st.markdown("### 📈 Interpolation Graph")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Generate smooth curve
                x_min = min(x_points) - 0.5
                x_max = max(x_points) + 0.5
                x_smooth = np.linspace(x_min, x_max, 500)
                y_smooth = [result['polynomial'](xi) for xi in x_smooth]
                
                # Plot polynomial
                ax.plot(x_smooth, y_smooth, 'b-', linewidth=2.5, label='Lagrange Polynomial')
                ax.plot(x_points, y_points, 'ro', markersize=12, label='Data Points', zorder=5)
                ax.plot(eval_x, eval_result, 'g*', markersize=20, label=f'P({eval_x}) = {eval_result:.4f}', zorder=10)
                
                # Add point labels
                for i, (xi, yi) in enumerate(zip(x_points, y_points)):
                    ax.annotate(f'({xi:.2f}, {yi:.2f})', 
                               xy=(xi, yi), 
                               xytext=(10, 10), 
                               textcoords='offset points',
                               fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
                
                ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
                ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=11, loc='best')
                ax.set_xlabel('x', fontsize=13, fontweight='bold')
                ax.set_ylabel('P(x)', fontsize=13, fontweight='bold')
                ax.set_title(f'Lagrange Interpolation (Degree {degree})', fontsize=15, fontweight='bold')
                
                st.pyplot(fig)
                
                # Verification table
                st.markdown("### ✅ Verification at Data Points")
                verification_data = []
                for i, point in enumerate(result['points']):
                    p_val = result['polynomial'](point['x'])
                    error = abs(p_val - point['y'])
                    verification_data.append({
                        'Point #': i + 1,
                        'x': f"{point['x']:.4f}",
                        'Actual y': f"{point['y']:.4f}",
                        'P(x)': f"{p_val:.4f}",
                        'Error': f"{error:.2e}"
                    })
                
                ver_df = pd.DataFrame(verification_data)
                st.dataframe(ver_df, use_container_width=True, hide_index=True)
                
                # Check if perfect fit
                max_error = max([abs(result['polynomial'](point['x']) - point['y']) for point in result['points']])
                if max_error < 1e-10:
                    st.success("✅ Perfect fit! Polynomial passes through all data points.")
                else:
                    st.info(f"ℹ️ Maximum error at data points: {max_error:.2e}")
        
        else:
            st.error(f"❌ {result['message']}")
    
    else:
        # Welcome screen for Lagrange
        st.info("👈 **Get Started:** Enter your data points and click INTERPOLATE")
        
        st.markdown("### 📚 About Lagrange Interpolation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **What is Lagrange Interpolation?**
            
            Lagrange interpolation constructs a polynomial that passes through a given set of points. 
            The polynomial has the minimum degree required to pass through all points.
            
            **Formula:**
            """)
            st.latex(r"P(x) = \sum_{i=0}^{n} y_i \cdot L_i(x)")
            st.latex(r"L_i(x) = \prod_{\substack{j=0\\j \neq i}}^{n} \frac{x - x_j}{x_i - x_j}")
        
        with col2:
            st.markdown("""
            **Degree Guidelines:**
            
            - **Degree 1 (Linear):** 2 points → straight line
            - **Degree 2 (Quadratic):** 3 points → parabola
            - **Degree 3 (Cubic):** 4 points → cubic curve
            
            **Applications:**
            - Data fitting and curve approximation
            - Numerical integration (Newton-Cotes formulas)
            - Missing data estimation
            - Function approximation
            """)
        
        st.markdown("---")
        st.markdown("### 💡 Quick Example")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Linear (Degree 1):**")
            st.code("Points: (0, 1), (1, 3)\nResult: P(x) = 2x + 1", language="python")
        
        with col2:
            st.markdown("**Quadratic (Degree 2):**")
            st.code("Points: (0, 1), (1, 3), (2, 2)\nResult: Parabola through all 3 points", language="python")

        st.markdown("---")
# ============================================================================
# DIVIDED DIFFERENCE INTERPOLATION SECTION
# ============================================================================
elif problem_type == "🔢 Divided Difference Interpolation":
        st.sidebar.markdown("### 🔢 Divided Difference Method")
        
        # Degree selection
        degree = st.sidebar.selectbox(
            "Polynomial Degree:",
            [1, 2, 3, 4],
            index=2,
            help="Select the degree of interpolating polynomial"
        )
        
        st.sidebar.info(f"💡 You need {degree + 1} data points for degree {degree} polynomial")
        
        # Number of points
        num_points = degree + 1
        
        st.sidebar.markdown(f"### 📍 Enter {num_points} Data Points")
        
        # Initialize session state for points
        if 'x_points_dd' not in st.session_state:
            st.session_state.x_points_dd = [0.0] * 5
        if 'y_points_dd' not in st.session_state:
            st.session_state.y_points_dd = [0.0] * 5
        
        # Default examples
        default_points_dd = {
            1: {  # degree 1 → 2 points
                'x': [8.0, 8.1],
                'y': [16.63553, 17.61549]
            },
            2: {  # degree 2 → 3 points
                'x': [8.0, 8.1, 8.3],
                'y': [16.63553, 17.61549, 17.56492]
            },
            3: {  # degree 3 → 4 points
                'x': [8.0, 8.1, 8.3, 8.6],
                'y': [16.63553, 17.61549, 17.56492, 18.50515]
            },
            4: {  # degree 4 → 5 points
                'x': [8.0, 8.1, 8.3, 8.6, 8.7],
                'y': [16.63553, 17.61549, 17.56492, 18.50515, 18.82091]
            }
        }

        
        # Quick example button
        if st.sidebar.button("📝 Load Example", use_container_width=True, key="load_dd_example"):
            st.session_state.x_points_dd = default_points_dd[degree]['x'] + [0.0] * (5 - len(default_points_dd[degree]['x']))
            st.session_state.y_points_dd = default_points_dd[degree]['y'] + [0.0] * (5 - len(default_points_dd[degree]['y']))
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Input points
        x_points_dd = []
        y_points_dd = []
        
        for i in range(num_points):
            st.sidebar.markdown(f"**Point {i + 1}:**")
            col1, col2 = st.sidebar.columns(2)
            
            x_val = col1.number_input(
                f"x{i}:",
                value=float(st.session_state.x_points_dd[i]) if i < len(st.session_state.x_points_dd) else 0.0,
                format="%.10f",
                key=f"x_dd_{i}"
            )
            y_val = col2.number_input(
                f"y{i}:",
                value=float(st.session_state.y_points_dd[i]) if i < len(st.session_state.y_points_dd) else 0.0,
                format="%.10f",
                key=f"y_dd_{i}"
            )
            
            x_points_dd.append(x_val)
            y_points_dd.append(y_val)
        
        # Update session state
        st.session_state.x_points_dd = x_points_dd + [0.0] * (5 - len(x_points_dd))
        st.session_state.y_points_dd = y_points_dd + [0.0] * (5 - len(y_points_dd))
        
        st.sidebar.markdown("---")
        
        # Evaluation point
        st.sidebar.markdown("### 🎯 Evaluation")
        eval_x_dd = st.sidebar.number_input(
            "Evaluate P(x) at:",
            value=0.5,
            format="%.10f",
            help="Enter x value to evaluate the polynomial",
            key="eval_x_dd"
        )
        
        # Display options
        st.sidebar.markdown("### 📊 Display")
        show_table_dd = st.sidebar.checkbox("Show Divided Difference Table", value=True, key="show_table_dd")
        show_graph_dd = st.sidebar.checkbox("Show Graph", value=True, key="show_graph_dd")
        show_steps_dd = st.sidebar.checkbox("Show Step-by-Step Solution", value=True, key="show_steps_dd")
        
        st.sidebar.markdown("---")
        calculate_dd = st.sidebar.button("🚀 INTERPOLATE", type="primary", use_container_width=True, key="calc_dd")
        
        # Main content
        if calculate_dd:
            # Validate unique x values
            if len(set(x_points_dd)) != len(x_points_dd):
                st.error("❌ Error: All x values must be unique!")
                st.stop()
            
            # Compute interpolation
            with st.spinner("Calculating divided difference interpolation..."):
                result_dd = divided_difference(x_points_dd, y_points_dd, degree=degree)
            
            if result_dd['success']:
                st.success(f"✅ {result_dd['message']}")
                
                # Display Newton polynomial form
                st.markdown("### 📐 Newton's Divided Difference Polynomial")
                
                newton_poly_str = format_newton_polynomial(result_dd['coefficients'], result_dd['x_points'])
                st.code(newton_poly_str, language="text")
                
                st.markdown("---")
                
                # Step-by-step solution
                if show_steps_dd:
                    st.markdown("### 📝 Detailed Step-by-Step Solution")
                    
                    # Step 1: Understanding the method
                    with st.expander("📖 **Step 1: Understanding Newton's Divided Difference Method**", expanded=True):
                        st.markdown("""
                        **Newton's Divided Difference Form:**
                        
                        The polynomial is constructed as:
                        """)
                        st.latex(r"P(x) = a_0 + a_1(x-x_0) + a_2(x-x_0)(x-x_1) + ... + a_n(x-x_0)(x-x_1)...(x-x_{n-1})")
                        
                        st.markdown("""
                        **Advantages over Lagrange:**
                        - More efficient when adding new data points
                        - Easier to compute for many points
                        - Divided difference table shows the structure clearly
                        """)
                    
                    # Step 2: Build divided difference table
                    with st.expander("🔧 **Step 2: Building the Divided Difference Table**", expanded=True):
                        st.markdown("""
                        We build a table where:
                        - **Column 0:** Original y-values (f[x_i])
                        - **Column 1:** First divided differences f[x_i, x_{i+1}]
                        - **Column 2:** Second divided differences f[x_i, x_{i+1}, x_{i+2}]
                        - And so on...
                        
                        **Formula for divided differences:**
                        """)
                        st.latex(r"f[x_i, x_{i+1}, ..., x_{i+k}] = \frac{f[x_{i+1}, ..., x_{i+k}] - f[x_i, ..., x_{i+k-1}]}{x_{i+k} - x_i}")
                        
                        # Show calculations
                        n = len(x_points_dd)
                        dd_table = result_dd['divided_diff_table']
                        
                        st.markdown("#### Calculation Steps:")
                        
                        # Show first differences
                        st.markdown("**First Divided Differences:**")
                        for i in range(n - 1):
                            numerator = dd_table[i+1][0] - dd_table[i][0]
                            denominator = x_points_dd[i+1] - x_points_dd[i]
                            result_val = dd_table[i][1]
                            st.code(f"f[x{i}, x{i+1}] = ({format_max_precision(dd_table[i+1][0])} - {format_max_precision(dd_table[i][0])}) / ({format_max_precision(x_points_dd[i+1])} - {format_max_precision(x_points_dd[i])}) = {format_max_precision(result_val)}")
                        
                        # Show second differences if applicable
                        if n > 2:
                            st.markdown("**Second Divided Differences:**")
                            for i in range(n - 2):
                                numerator = dd_table[i+1][1] - dd_table[i][1]
                                denominator = x_points_dd[i+2] - x_points_dd[i]
                                result_val = dd_table[i][2]
                                st.code(f"f[x{i}, x{i+1}, x{i+2}] = ({format_max_precision(dd_table[i+1][1])} - {format_max_precision(dd_table[i][1])}) / ({format_max_precision(x_points_dd[i+2])} - {format_max_precision(x_points_dd[i])}) = {format_max_precision(result_val)}")
                        
                        # Show third differences if applicable
                        if n > 3:
                            st.markdown("**Third Divided Differences:**")
                            for i in range(n - 3):
                                numerator = dd_table[i+1][2] - dd_table[i][2]
                                denominator = x_points_dd[i+3] - x_points_dd[i]
                                result_val = dd_table[i][3]
                                st.code(f"f[x{i}, x{i+1}, x{i+2}, x{i+3}] = ({format_max_precision(dd_table[i+1][2])} - {format_max_precision(dd_table[i][2])}) / ({format_max_precision(x_points_dd[i+3])} - {format_max_precision(x_points_dd[i])}) = {format_max_precision(result_val)}")
                        
                        # Show fourth differences if applicable
                        if n > 4:
                            st.markdown("**Fourth Divided Differences:**")
                            for i in range(n - 4):
                                numerator = dd_table[i+1][3] - dd_table[i][3]
                                denominator = x_points_dd[i+4] - x_points_dd[i]
                                result_val = dd_table[i][4]
                                st.code(f"f[x{i}, x{i+1}, x{i+2}, x{i+3}, x{i+4}] = ({format_max_precision(dd_table[i+1][3])} - {format_max_precision(dd_table[i][3])}) / ({format_max_precision(x_points_dd[i+4])} - {format_max_precision(x_points_dd[i])}) = {format_max_precision(result_val)}")
                    
                    # Step 3: Extract coefficients
                    with st.expander("📊 **Step 3: Extract Newton Coefficients**", expanded=True):
                        st.markdown("The coefficients for Newton's form are taken from the **first row** (or diagonal) of the divided difference table:")
                        
                        coeff_data = []
                        for i, coeff in enumerate(result_dd['coefficients']):
                            if i == 0:
                                term = f"a₀"
                                corresponds = "f[x₀]"
                            elif i == 1:
                                term = f"a₁"
                                corresponds = "f[x₀, x₁]"
                            elif i == 2:
                                term = f"a₂"
                                corresponds = "f[x₀, x₁, x₂]"
                            else:
                                term = f"a₃"
                                corresponds = f"f[x₀, x₁, ..., x₂]"
                            
                            coeff_data.append({
                                'Coefficient': term,
                                'Value': format_max_precision(coeff),
                                'Corresponds to': corresponds
                            })
                        
                        st.table(pd.DataFrame(coeff_data))
                    
                    # Step 4: Build polynomial
                    with st.expander("🔨 **Step 4: Construct the Polynomial**", expanded=True):
                        st.markdown("Using the coefficients, we build the Newton polynomial:")
                        
                        st.code(newton_poly_str, language="text")
                        
                        st.markdown("**Breaking it down:**")
                        for i, coeff in enumerate(result_dd['coefficients']):
                            if i == 0:
                                st.write(f"- Term {i+1}: `{format_max_precision(coeff)}` (constant term)")
                            else:
                                factors = " × ".join([f"(x - {format_max_precision(result_dd['x_points'][j])})" for j in range(i)])
                                st.write(f"- Term {i+1}: `{format_max_precision(coeff)} × {factors}`")
                    
                    # Step 5: Evaluate
                    with st.expander(f"🎯 **Step 5: Evaluate at x = {eval_x_dd}**", expanded=True):
                        st.markdown(f"Let's evaluate P({eval_x_dd}) step by step using nested multiplication:")
                        
                        # Show Horner's method calculation
                        st.markdown("**Using Horner's Method (efficient evaluation):**")
                        
                        eval_steps = []
                        n = len(result_dd['coefficients'])
                        value = result_dd['coefficients'][n - 1]
                        
                        eval_steps.append(f"Start with last coefficient: {format_max_precision(value)}")
                        
                        for i in range(n - 2, -1, -1):
                            old_value = value
                            value = value * (eval_x_dd - result_dd['x_points'][i]) + result_dd['coefficients'][i]
                            eval_steps.append(f"Step {n-i}: {format_max_precision(old_value)} × ({format_max_precision(eval_x_dd)} - {format_max_precision(result_dd['x_points'][i])}) + {format_max_precision(result_dd['coefficients'][i])} = {format_max_precision(value)}")
                        
                        for step in eval_steps:
                            st.code(step)
                        
                        st.success(f"**Final Result:** P({format_max_precision(eval_x_dd)}) = {format_max_precision(value)}")
                
                st.markdown("---")
                
                # Divided Difference Table
                if show_table_dd:
                    st.markdown("### 📊 Divided Difference Table")
                    
                    # Create formatted table
                    n = len(x_points_dd)
                    dd_table = result_dd['divided_diff_table']
                    
                    # Build table with proper headers
                    table_data = []
                    for i in range(n):
                        row = {'i': i, 'x_i': format_max_precision(x_points_dd[i]), 'f[x_i]': format_max_precision(dd_table[i][0])}
                        
                        for j in range(1, n):
                            if i + j < n:
                                if j == 1:
                                    row[f'f[x_i,x_i+1]'] = format_max_precision(dd_table[i][j])
                                elif j == 2:
                                    row[f'f[x_i,...,x_i+2]'] = format_max_precision(dd_table[i][j])
                                else:
                                    row[f'f[x_i,...,x_i+{j}]'] = format_max_precision(dd_table[i][j])
                            else:
                                if j == 1:
                                    row[f'f[x_i,x_i+1]'] = ""
                                elif j == 2:
                                    row[f'f[x_i,...,x_i+2]'] = ""
                                else:
                                    row[f'f[x_i,...,x_i+{j}]'] = ""
                        
                        table_data.append(row)
                    
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
                    
                    st.info("💡 **Reading the table:** The first row contains all coefficients for Newton's polynomial!")
                
                # Metrics
                st.markdown("---")
                eval_result_dd = result_dd['polynomial'](eval_x_dd)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Polynomial Degree", result_dd['degree'])
                col2.metric(f"P({format_max_precision(eval_x_dd)})", format_max_precision(eval_result_dd))
                col3.metric("Data Points Used", len(result_dd['points']))
                
                st.markdown("---")
                
                # Data points table
                st.markdown("### 📍 Input Data Points")
                points_df_dd = pd.DataFrame(result_dd['points'])
                # Format to maximum precision
                for col in points_df_dd.select_dtypes(include=[np.number]).columns:
                    points_df_dd[col] = points_df_dd[col].apply(lambda x: format_max_precision(x))
                st.dataframe(points_df_dd, use_container_width=True, hide_index=True)
                
                # Graph
                if show_graph_dd:
                    st.markdown("---")
                    st.markdown("### 📈 Interpolation Graph")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Generate smooth curve
                    x_min = min(x_points_dd) - 0.5
                    x_max = max(x_points_dd) + 0.5
                    x_smooth = np.linspace(x_min, x_max, 500)
                    y_smooth = [result_dd['polynomial'](xi) for xi in x_smooth]
                    
                    # Plot polynomial
                    ax.plot(x_smooth, y_smooth, 'g-', linewidth=2.5, label='Newton Polynomial')
                    ax.plot(x_points_dd, y_points_dd, 'ro', markersize=12, label='Data Points', zorder=5)
                    ax.plot(eval_x_dd, eval_result_dd, 'b*', markersize=20, label=f'P({eval_x_dd}) = {eval_result_dd:.4f}', zorder=10)
                    
                    # Add point labels
                    for i, (xi, yi) in enumerate(zip(x_points_dd, y_points_dd)):
                        ax.annotate(f'({xi:.2f}, {yi:.2f})', 
                                xy=(xi, yi), 
                                xytext=(10, 10), 
                                textcoords='offset points',
                                fontsize=9,
                                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
                    
                    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
                    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=11, loc='best')
                    ax.set_xlabel('x', fontsize=13, fontweight='bold')
                    ax.set_ylabel('P(x)', fontsize=13, fontweight='bold')
                    ax.set_title(f'Newton\'s Divided Difference Interpolation (Degree {degree})', fontsize=15, fontweight='bold')
                    
                    st.pyplot(fig)
                    
                    # Verification
                    st.markdown("### ✅ Verification at Data Points")
                    verification_data_dd = []
                    for i, point in enumerate(result_dd['points']):
                        p_val = result_dd['polynomial'](point['x'])
                        error = abs(p_val - point['y'])
                        verification_data_dd.append({
                            'Point #': i + 1,
                            'x': f"{point['x']:.4f}",
                            'Actual y': f"{point['y']:.4f}",
                            'P(x)': f"{p_val:.4f}",
                            'Error': f"{error:.2e}"
                        })
                    
                    ver_df_dd = pd.DataFrame(verification_data_dd)
                    st.dataframe(ver_df_dd, use_container_width=True, hide_index=True)
                    
                    max_error_dd = max([abs(result_dd['polynomial'](point['x']) - point['y']) for point in result_dd['points']])
                    if max_error_dd < 1e-10:
                        st.success("✅ Perfect fit! Polynomial passes through all data points.")
                    else:
                        st.info(f"ℹ️ Maximum error at data points: {max_error_dd:.2e}")
            
            else:
                st.error(f"❌ {result_dd['message']}")
        
        else:
            # Welcome screen
            st.info("👈 **Get Started:** Enter your data points and click INTERPOLATE")
            
            st.markdown("### 📚 About Newton's Divided Difference Method")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **What is Divided Difference Interpolation?**
                
                Newton's divided difference method constructs an interpolating polynomial using divided differences. It produces the same polynomial as Lagrange but in a different form.
                
                **Newton's Form:**
                """)
                st.latex(r"P(x) = a_0 + a_1(x-x_0) + a_2(x-x_0)(x-x_1) + ...")
                
                st.markdown("""
                **Divided Difference Formula:**
                """)
                st.latex(r"f[x_i, x_{i+1}, ..., x_{i+k}] = \frac{f[x_{i+1}, ..., x_{i+k}] - f[x_i, ..., x_{i+k-1}]}{x_{i+k} - x_i}")
            
            with col2:
                st.markdown("""
                **Advantages:**
                
                - ✅ **Incremental:** Easy to add new points
                - ✅ **Efficient:** Less computation than Lagrange for multiple points
                - ✅ **Clear structure:** Divided difference table shows calculations
                - ✅ **Same result:** Produces identical polynomial to Lagrange
                
                **Applications:**
                - Dynamic data fitting (adding points over time)
                - Numerical differentiation
                - Approximation theory
                - Computer graphics
                """)
            
            st.markdown("---")
            st.markdown("### 💡 Quick Example")
            st.code("""
    Example: Points (0,1), (1,3), (2,2)

    Divided Difference Table:
    i  x_i  f[x_i]  f[x_i,x_i+1]  f[x_i,x_i+1,x_i+2]
    0  0    1       2.0           -1.5
    1  1    3       -1.0
    2  2    2

    Newton's Form:
    P(x) = 1 + 2(x-0) + (-1.5)(x-0)(x-1)
        = 1 + 2x - 1.5x² + 1.5x
        = -1.5x² + 3.5x + 1
            """)

           # ============================================================================
# JACOBI METHOD SECTION (Linear Systems)
# ============================================================================
elif problem_type == "🔧 Linear Systems (Jacobi)":
    st.sidebar.markdown("### 🔧 Jacobi Iterative Method")
    
    # System size selection
    system_size = st.sidebar.selectbox(
        "System Size (n×n):",
        [2, 3, 4, 5],
        index=1,
        help="Select the size of your linear system"
    )
    
    st.sidebar.info(f"💡 You'll enter a {system_size}×{system_size} matrix A and a {system_size}×1 vector b")
    
    # Initialize session state for matrix and vector
    if 'matrix_A' not in st.session_state:
        st.session_state.matrix_A = [[0.0] * system_size for _ in range(system_size)]
    if 'vector_b' not in st.session_state:
        st.session_state.vector_b = [0.0] * system_size
    if 'initial_guess' not in st.session_state:
        st.session_state.initial_guess = [0.0] * system_size
    
    # Adjust sizes if system_size changed
    if len(st.session_state.matrix_A) != system_size:
        st.session_state.matrix_A = [[0.0] * system_size for _ in range(system_size)]
        st.session_state.vector_b = [0.0] * system_size
        st.session_state.initial_guess = [0.0] * system_size
    
    # Example systems
    examples = {
        2: {
            'name': 'Simple 2×2',
            'A': [[4, 1], [1, 3]],
            'b': [1, 2],
            'x0': [0, 0]
        },
        3: {
            'name': 'Diagonally Dominant 3×3',
            'A': [[10, -1, 2], [-1, 11, -1], [2, -1, 10]],
            'b': [6, 25, -11],
            'x0': [0, 0, 0]
        },
        4: {
            'name': 'Sparse 4×4',
            'A': [[10, 1, 0, 0], [1, 10, 1, 0], [0, 1, 10, 1], [0, 0, 1, 10]],
            'b': [12, 13, 13, 12],
            'x0': [0, 0, 0, 0]
        },
        5: {
            'name': 'Tridiagonal 5×5',
            'A': [[10, 1, 0, 0, 0], [1, 10, 1, 0, 0], [0, 1, 10, 1, 0], [0, 0, 1, 10, 1], [0, 0, 0, 1, 10]],
            'b': [12, 13, 14, 13, 12],
            'x0': [0, 0, 0, 0, 0]
        }
    }
    
    # Quick example button
    if st.sidebar.button("📝 Load Example System", use_container_width=True, key="load_jacobi_example"):
        example = examples[system_size]
        st.session_state.matrix_A = [row[:] for row in example['A']]
        st.session_state.vector_b = example['b'][:]
        st.session_state.initial_guess = example['x0'][:]
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Matrix A input
    st.sidebar.markdown(f"### 📊 Matrix A ({system_size}×{system_size})")
    
    matrix_A = []
    for i in range(system_size):
        st.sidebar.markdown(f"**Row {i+1}:**")
        row_cols = st.sidebar.columns(system_size)
        row = []
        for j in range(system_size):
            val = row_cols[j].number_input(
                f"a{i+1}{j+1}",
                value=float(st.session_state.matrix_A[i][j]),
                format="%.4f",
                key=f"a_{i}_{j}",
                label_visibility="collapsed"
            )
            row.append(val)
        matrix_A.append(row)
    
    st.session_state.matrix_A = matrix_A
    
    st.sidebar.markdown("---")
    
    # Vector b input
    st.sidebar.markdown(f"### 📍 Vector b ({system_size}×1)")
    vector_b = []
    for i in range(system_size):
        val = st.sidebar.number_input(
            f"b{i+1}:",
            value=float(st.session_state.vector_b[i]),
            format="%.4f",
            key=f"b_{i}"
        )
        vector_b.append(val)
    
    st.session_state.vector_b = vector_b
    
    st.sidebar.markdown("---")
    
    # Initial guess
    st.sidebar.markdown("### 🎯 Initial Guess x₀")
    use_zero_guess = st.sidebar.checkbox("Use zero vector", value=True, key="use_zero_guess")
    
    if not use_zero_guess:
        initial_guess = []
        for i in range(system_size):
            val = st.sidebar.number_input(
                f"x₀[{i+1}]:",
                value=float(st.session_state.initial_guess[i]),
                format="%.4f",
                key=f"x0_{i}"
            )
            initial_guess.append(val)
        st.session_state.initial_guess = initial_guess
    else:
        initial_guess = None
    
    st.sidebar.markdown("---")
    
    # Method parameters
    st.sidebar.markdown("### ⚙️ Method Parameters")
    tolerance_jacobi = st.sidebar.number_input(
        "Tolerance:",
        value=1e-6,
        format="%.1e",
        min_value=1e-12,
        key="tol_jacobi"
    )
    max_iter_jacobi = st.sidebar.number_input(
        "Max Iterations:",
        value=100,
        min_value=1,
        max_value=1000,
        key="max_iter_jacobi"
    )
    
    # Display options
    st.sidebar.markdown("### 📊 Display")
    show_steps_jacobi = st.sidebar.checkbox("Show Step-by-Step", value=True, key="show_steps_jacobi")
    show_convergence_plot = st.sidebar.checkbox("Show Convergence Plot", value=True, key="show_conv_plot")
    show_dominance_check = st.sidebar.checkbox("Show Diagonal Dominance Check", value=True, key="show_dom_check")
    number_format_jacobi = st.sidebar.radio("Number Format:", ["Decimal", "Scientific"], horizontal=True, key="num_format_jacobi")
    
    st.sidebar.markdown("---")
    calculate_jacobi = st.sidebar.button("🚀 SOLVE SYSTEM", type="primary", use_container_width=True, key="calc_jacobi")
    
    # Main content
    if calculate_jacobi:
        st.session_state.has_interacted = True
        
        # Convert to numpy arrays
        A = np.array(matrix_A, dtype=float)
        b = np.array(vector_b, dtype=float)
        x0 = np.array(initial_guess) if initial_guess is not None else None
        
        # Display the system
        st.markdown("### 📐 Linear System: Ax = b")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("**Matrix A:**")
            st.code(format_matrix(A), language="text")
        
        with col2:
            st.markdown("**Vector b:**")
            st.code(format_vector(b), language="text")
        
        with col3:
            st.markdown("**Initial x₀:**")
            if x0 is not None:
                st.code(format_vector(x0), language="text")
            else:
                st.code(format_vector(np.zeros(system_size)), language="text")
        
        st.markdown("---")
        
        # Add this code section after displaying the Linear System (after st.markdown("---"))
        # and before the Diagonal Dominance Check

        # Display Jacobi Iteration Equations
        st.markdown("### 📐 Jacobi Iteration Equations")
        st.markdown("The Jacobi method solves each variable using:")

        # Display general formula
        st.latex(r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j \neq i} a_{ij} x_j^{(k)} \right)")

        st.markdown("**For this system:**")

        # Create columns for better layout if system is small
        if system_size <= 3:
            cols = st.columns(system_size)
        else:
            cols = None

        for i in range(system_size):
            # Build the equation string for LaTeX
            numerator_parts = [f"{b[i]:.4g}"]  # Start with b_i
            
            # Collect all off-diagonal terms
            for j in range(system_size):
                if i != j and A[i, j] != 0:
                    coeff = A[i, j]
                    if coeff > 0:
                        numerator_parts.append(f"- {coeff:.4g} x_{{{j+1}}}^{{(k)}}")
                    else:
                        numerator_parts.append(f"+ {abs(coeff):.4g} x_{{{j+1}}}^{{(k)}}")
            
            # Construct the full equation
            numerator = " ".join(numerator_parts)
            denominator = f"{A[i, i]:.4g}"
            
            equation = f"x_{{{i+1}}}^{{(k+1)}} = \\frac{{{numerator}}}{{{denominator}}}"
            
            # Display in columns if system is small, otherwise stack vertically
            if cols and i < len(cols):
                with cols[i]:
                    st.latex(equation)
            else:
                st.latex(equation)

        st.markdown("---")
        # Check diagonal dominance first
        if show_dominance_check:
            st.markdown("### 🔍 Diagonal Dominance Check")
            
            is_dominant, details = check_diagonal_dominance(A)
            
            if is_dominant:
                st.success("✅ Matrix is **strictly diagonally dominant**. Jacobi method is **guaranteed to converge**!")
            else:
                st.warning("⚠️ Matrix is **NOT strictly diagonally dominant**. Convergence is **not guaranteed** but may still occur.")
            
            # Show details table
            dom_df = pd.DataFrame(details)
            st.dataframe(dom_df, use_container_width=True, hide_index=True)
            
            # Calculate spectral radius
            try:
                spectral_rad = calculate_spectral_radius(A)
                st.markdown(f"**Spectral Radius ρ(T_J):** {spectral_rad:.6f}")
                
                if spectral_rad < 1:
                    st.success(f"✅ ρ(T_J) = {spectral_rad:.6f} < 1 → Method **will converge**")
                else:
                    st.error(f"❌ ρ(T_J) = {spectral_rad:.6f} ≥ 1 → Method **may not converge**")
            except:
                st.info("ℹ️ Could not calculate spectral radius")
            
            st.markdown("---")
        
        # Solve using Jacobi
        with st.spinner("Solving system using Jacobi method..."):
            result_jacobi = jacobi_method(A, b, x0=x0, tol=tolerance_jacobi, max_iter=max_iter_jacobi)
        
        if not result_jacobi['success']:
            st.error(f"❌ {result_jacobi['message']}")
            st.stop()
        
        # Display result
        if result_jacobi['converged']:
            st.success(result_jacobi['message'])
        else:
            st.warning(result_jacobi['message'])
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Converged", "✅ Yes" if result_jacobi['converged'] else "❌ No")
        col2.metric("Iterations", len(result_jacobi['iterations']) - 1)
        col3.metric("Final Error", f"{result_jacobi['final_error']:.2e}")
        
        # Calculate final residual
        final_residual = np.linalg.norm(np.dot(A, result_jacobi['solution']) - b, ord=np.inf)
        col4.metric("Residual ||Ax-b||", f"{final_residual:.2e}")
        
        st.markdown("---")
        
        # Display solution
        st.markdown("### ✅ Solution Vector x")
        
        sol_col1, sol_col2 = st.columns(2)
        
        with sol_col1:
            st.markdown("**Vector Form:**")
            st.code(format_vector(result_jacobi['solution']), language="text")
        
        with sol_col2:
            st.markdown("**Component Form:**")
            for i, val in enumerate(result_jacobi['solution']):
                if number_format_jacobi == "Scientific":
                    st.code(f"x{i+1} = {val:.6e}")
                else:
                    st.code(f"x{i+1} = {val:.8f}")
        
        st.markdown("---")
        
        # Verification
        st.markdown("### 🔬 Verification: Ax = b")
        
        Ax = np.dot(A, result_jacobi['solution'])
        
        ver_col1, ver_col2, ver_col3 = st.columns(3)
        
        with ver_col1:
            st.markdown("**Ax (computed):**")
            st.code(format_vector(Ax), language="text")
        
        with ver_col2:
            st.markdown("**b (expected):**")
            st.code(format_vector(b), language="text")
        
        with ver_col3:
            st.markdown("**Difference (Ax - b):**")
            difference = Ax - b
            st.code(format_vector(difference), language="text")
        
        st.markdown("---")
        
        # Step-by-step solution
        if show_steps_jacobi:
            st.markdown("### 📝 Detailed Step-by-Step Solution")
            
            # Step 1: Understanding
            with st.expander("📖 **Step 1: Understanding Jacobi Method**", expanded=True):
                st.markdown("""
                The **Jacobi method** is an iterative algorithm for solving linear systems **Ax = b**.
                
                #### Algorithm:
                1. Decompose matrix A = D + R, where:
                   - **D** = diagonal part of A
                   - **R** = remainder (off-diagonal elements)
                
                2. Iteration formula:
                """)
                st.latex(r"x^{(k+1)} = D^{-1}(b - Rx^{(k)})")
                
                st.markdown("Or component-wise:")
                st.latex(r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j \neq i} a_{ij} x_j^{(k)} \right)")
                
                st.markdown("""
                #### Convergence:
                - **Guaranteed** if A is strictly diagonally dominant
                - **May converge** for other matrices if spectral radius ρ(T_J) < 1
                """)
            
            # Step 2: Matrix decomposition
            with st.expander("🔧 **Step 2: Matrix Decomposition (A = D + R)**", expanded=False):
                D = np.diag(np.diag(A))
                R = A - D
                
                st.markdown("**Diagonal Matrix D:**")
                st.code(format_matrix(D), language="text")
                
                st.markdown("**Remainder Matrix R:**")
                st.code(format_matrix(R), language="text")
                
                st.markdown("**Inverse of D (D⁻¹):**")
                D_inv = np.diag(1.0 / np.diag(A))
                st.code(format_matrix(D_inv), language="text")
            
            # Step 3: Iteration formula for each component
            with st.expander("📐 **Step 3: Component-wise Iteration Formulas**", expanded=False):
                st.markdown("For each variable xᵢ, we use:")
                
                for i in range(system_size):
                    st.markdown(f"**Variable x{i+1}:**")
                    
                    # Build formula string
                    terms = []
                    for j in range(system_size):
                        if i != j:
                            if A[i, j] != 0:
                                terms.append(f"{A[i,j]:.4f}·x{j+1}")
                    
                    sum_str = " - ".join(terms) if terms else "0"
                    
                    st.latex(f"x_{{{i+1}}}^{{(k+1)}} = \\frac{{1}}{{{A[i,i]:.4f}}} \\left( {b[i]:.4f} - ({sum_str}) \\right)")
            
            # Step 4: Show first few iterations in detail
            with st.expander("🔄 **Step 4: Iteration Details (First 3 Iterations)**", expanded=False):
                for iter_idx in range(min(4, len(result_jacobi['iterations']))):
                    iter_data = result_jacobi['iterations'][iter_idx]
                    
                    st.markdown(f"#### Iteration {iter_data['iteration']}")
                    
                    if iter_data['iteration'] == 0:
                        st.markdown("**Initial guess:**")
                        st.code(format_vector(iter_data['x']), language="text")
                    else:
                        st.markdown("**Calculations:**")
                        
                        # Show calculation for each component
                        x_prev = result_jacobi['iterations'][iter_idx - 1]['x']
                        
                        for i in range(system_size):
                            sum_val = 0.0
                            calc_parts = []
                            
                            for j in range(system_size):
                                if i != j:
                                    sum_val += A[i, j] * x_prev[j]
                                    if A[i, j] != 0:
                                        calc_parts.append(f"{A[i,j]:.4f}×{x_prev[j]:.4f}")
                            
                            sum_str = " + ".join(calc_parts) if calc_parts else "0"
                            result_val = (b[i] - sum_val) / A[i, i]
                            
                            st.code(f"x{i+1} = ({b[i]:.4f} - ({sum_str})) / {A[i,i]:.4f} = {result_val:.6f}")
                        
                        st.markdown(f"**Updated x:**")
                        st.code(format_vector(iter_data['x']), language="text")
                    
                    st.markdown(f"**Error:** {iter_data['error']:.6e}")
                    st.markdown(f"**Residual:** {iter_data['residual']:.6e}")
                    st.markdown("---")
        
        # Iteration table
        st.markdown("### 📊 Complete Iteration History")
        
        # Prepare dataframe
        iter_display = []
        for iter_data in result_jacobi['iterations']:
            row = {'Iteration': iter_data['iteration']}
            
            # Add components
            for i in range(system_size):
                if number_format_jacobi == "Scientific":
                    row[f'x{i+1}'] = f"{iter_data[f'x{i+1}']:.6e}"
                else:
                    row[f'x{i+1}'] = f"{iter_data[f'x{i+1}']:.8f}"
            
            row['Error'] = f"{iter_data['error']:.6e}"
            row['Residual'] = f"{iter_data['residual']:.6e}"
            
            iter_display.append(row)
        
        iter_df = pd.DataFrame(iter_display)
        st.dataframe(iter_df, use_container_width=True, height=400)
        
        # Download button
        csv = pd.DataFrame(result_jacobi['iterations']).to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "jacobi_iterations.csv", "text/csv")
        
        # Convergence plot
        if show_convergence_plot and len(result_jacobi['iterations']) > 1:
            st.markdown("---")
            st.markdown("### 📈 Convergence Analysis")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Error plot
            iterations = [it['iteration'] for it in result_jacobi['iterations']]
            errors = [it['error'] for it in result_jacobi['iterations']]
            residuals = [it['residual'] for it in result_jacobi['iterations']]
            
            ax1.semilogy(iterations, errors, 'b-o', linewidth=2, markersize=6, label='Error ||x^(k+1) - x^(k)||')
            ax1.axhline(y=tolerance_jacobi, color='red', linestyle='--', linewidth=1.5, label=f'Tolerance = {tolerance_jacobi:.1e}')
            ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Error (log scale)', fontsize=12, fontweight='bold')
            ax1.set_title('Convergence: Error vs Iteration', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3, which='both')
            
            # Residual plot
            ax2.semilogy(iterations, residuals, 'g-s', linewidth=2, markersize=6, label='Residual ||Ax - b||')
            ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Residual (log scale)', fontsize=12, fontweight='bold')
            ax2.set_title('Convergence: Residual vs Iteration', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3, which='both')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Component-wise convergence
            st.markdown("#### 📊 Component-wise Convergence")
            
            fig2, ax3 = plt.subplots(figsize=(12, 6))
            
            for i in range(system_size):
                component_vals = [it[f'x{i+1}'] for it in result_jacobi['iterations']]
                ax3.plot(iterations, component_vals, '-o', linewidth=2, markersize=5, label=f'x{i+1}')
            
            ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Value', fontsize=12, fontweight='bold')
            ax3.set_title('Evolution of Solution Components', fontsize=14, fontweight='bold')
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
            
            st.pyplot(fig2)
    
    else:
        # Welcome screen
        st.info("👈 **Get Started:** Enter your linear system Ax = b in the sidebar and click 'SOLVE SYSTEM'")
        
        st.markdown("### 📚 About Jacobi Iterative Method")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **What is the Jacobi Method?**
            
            The Jacobi method is an **iterative algorithm** for solving linear systems **Ax = b**. 
            It's particularly useful for **large sparse systems** where direct methods (like Gaussian elimination) are impractical.
            
            #### Algorithm:
            Starting from an initial guess x⁽⁰⁾, we iteratively update:
            """)
            st.latex(r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j \neq i} a_{ij} x_j^{(k)} \right)")
            
            st.markdown("""
            #### Key Features:
            - ✅ **Simple to implement**
            - ✅ **Parallelizable** (updates independent)
            - ✅ **Low memory usage**
            - ⚠️ **Slower convergence** than Gauss-Seidel
            """)
        
        with col2:
            st.markdown("""
            **Convergence Conditions:**
            
            The Jacobi method converges if:
            
            1. **Strictly Diagonally Dominant Matrix:**
            """)
            st.latex(r"|a_{ii}| > \sum_{j \neq i} |a_{ij}| \quad \forall i")
            
            st.markdown("""
            2. **Spectral Radius Condition:**
            """)
            st.latex(r"\rho(D^{-1}R) < 1")
            
            st.markdown("""
            where D is the diagonal part and R is the remainder.
            
            #### Applications:
            - Solving PDEs (Poisson, heat equation)
            - Circuit analysis
            - Structural analysis
            - Image processing
            - Machine learning (distributed optimization)
            """)
        
        st.markdown("---")
        st.markdown("### 💡 Example Systems")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Diagonally Dominant (Converges):**")
            st.code("""
A = [ 10  -1   2 ]    b = [  6 ]
    [ -1  11  -1 ]        [ 25 ]
    [  2  -1  10 ]        [-11 ]

Solution: x ≈ [1, 2, -1]
            """, language="text")
        
        with col2:
            st.markdown("**Not Dominant (May Not Converge):**")
            st.code("""
A = [ 2   3 ]    b = [ 5 ]
    [ 3   2 ]        [ 5 ]

May fail to converge!
Use Gauss-Seidel or direct methods.
            """, language="text")
        
        st.markdown("---")
        st.markdown("### 🎓 Convergence Theory")
        
        st.markdown("""
        **Error Bound:**
        
        If A is strictly diagonally dominant, the error after k iterations satisfies:
        """)
        st.latex(r"\|x^{(k)} - x^*\| \leq \left(\frac{\max_i \sum_{j \neq i} |a_{ij}|}{\min_i |a_{ii}|}\right)^k \|x^{(0)} - x^*\|")
        
        st.markdown("""
        **Rate of Convergence:**
        - Linear convergence with rate ρ(T_J)
        - Faster when matrix is "more" diagonally dominant
        - Convergence rate independent of initial guess
        """)

        # ============================================================================
# GAUSS-SEIDEL METHOD SECTION (Linear Systems)
# ============================================================================
# Add this after the Jacobi section (or as a separate option)

elif problem_type == "⚡ Linear Systems (Gauss-Seidel)":
    st.sidebar.markdown("### ⚡ Gauss-Seidel Iterative Method")
    
    # System size selection
    system_size = st.sidebar.selectbox(
        "System Size (n×n):",
        [2, 3, 4, 5],
        index=1,
        help="Select the size of your linear system"
    )
    
    st.sidebar.info(f"💡 You'll enter a {system_size}×{system_size} matrix A and a {system_size}×1 vector b")
    
    # Initialize session state for matrix and vector
    if 'matrix_A_gs' not in st.session_state:
        st.session_state.matrix_A_gs = [[0.0] * system_size for _ in range(system_size)]
    if 'vector_b_gs' not in st.session_state:
        st.session_state.vector_b_gs = [0.0] * system_size
    if 'initial_guess_gs' not in st.session_state:
        st.session_state.initial_guess_gs = [0.0] * system_size
    
    # Adjust sizes if system_size changed
    if len(st.session_state.matrix_A_gs) != system_size:
        st.session_state.matrix_A_gs = [[0.0] * system_size for _ in range(system_size)]
        st.session_state.vector_b_gs = [0.0] * system_size
        st.session_state.initial_guess_gs = [0.0] * system_size
    
    # Example systems
    examples_gs = {
        2: {
            'name': 'Simple 2×2',
            'A': [[4, 1], [1, 3]],
            'b': [1, 2],
            'x0': [0, 0]
        },
        3: {
            'name': 'Diagonally Dominant 3×3',
            'A': [[10, -1, 2], [-1, 11, -1], [2, -1, 10]],
            'b': [6, 25, -11],
            'x0': [0, 0, 0]
        },
        4: {
            'name': '4×4 System',
            'A': [[4, 1, -1, 1], [1, 4, -1, -1], [-1, -1, 5, 1], [1, -1, 1, 3]],
            'b': [-2, -1, 0, 1],
            'x0': [0, 0, 0, 0]
        },
        5: {
            'name': 'Tridiagonal 5×5',
            'A': [[10, 1, 0, 0, 0], [1, 10, 1, 0, 0], [0, 1, 10, 1, 0], [0, 0, 1, 10, 1], [0, 0, 0, 1, 10]],
            'b': [12, 13, 14, 13, 12],
            'x0': [0, 0, 0, 0, 0]
        }
    }
    
    # Quick example button
    if st.sidebar.button("📝 Load Example System", use_container_width=True, key="load_gs_example"):
        example = examples_gs[system_size]
        st.session_state.matrix_A_gs = [row[:] for row in example['A']]
        st.session_state.vector_b_gs = example['b'][:]
        st.session_state.initial_guess_gs = example['x0'][:]
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Matrix A input
    st.sidebar.markdown(f"### 📊 Matrix A ({system_size}×{system_size})")
    
    matrix_A_gs = []
    for i in range(system_size):
        st.sidebar.markdown(f"**Row {i+1}:**")
        row_cols = st.sidebar.columns(system_size)
        row = []
        for j in range(system_size):
            val = row_cols[j].number_input(
                f"a{i+1}{j+1}",
                value=float(st.session_state.matrix_A_gs[i][j]),
                format="%.4f",
                key=f"a_gs_{i}_{j}",
                label_visibility="collapsed"
            )
            row.append(val)
        matrix_A_gs.append(row)
    
    st.session_state.matrix_A_gs = matrix_A_gs
    
    st.sidebar.markdown("---")
    
    # Vector b input
    st.sidebar.markdown(f"### 📍 Vector b ({system_size}×1)")
    vector_b_gs = []
    for i in range(system_size):
        val = st.sidebar.number_input(
            f"b{i+1}:",
            value=float(st.session_state.vector_b_gs[i]),
            format="%.4f",
            key=f"b_gs_{i}"
        )
        vector_b_gs.append(val)
    
    st.session_state.vector_b_gs = vector_b_gs
    
    st.sidebar.markdown("---")
    
    # Initial guess
    st.sidebar.markdown("### 🎯 Initial Guess x₀")
    use_zero_guess_gs = st.sidebar.checkbox("Use zero vector", value=True, key="use_zero_guess_gs")
    
    if not use_zero_guess_gs:
        initial_guess_gs = []
        for i in range(system_size):
            val = st.sidebar.number_input(
                f"x₀[{i+1}]:",
                value=float(st.session_state.initial_guess_gs[i]),
                format="%.4f",
                key=f"x0_gs_{i}"
            )
            initial_guess_gs.append(val)
        st.session_state.initial_guess_gs = initial_guess_gs
    else:
        initial_guess_gs = None
    
    st.sidebar.markdown("---")
    
    # Method parameters
    st.sidebar.markdown("### ⚙️ Method Parameters")
    tolerance_gs = st.sidebar.number_input(
        "Tolerance:",
        value=1e-6,
        format="%.1e",
        min_value=1e-12,
        key="tol_gs"
    )
    max_iter_gs = st.sidebar.number_input(
        "Max Iterations:",
        value=100,
        min_value=1,
        max_value=1000,
        key="max_iter_gs"
    )
    
    # Display options
    st.sidebar.markdown("### 📊 Display")
    show_steps_gs = st.sidebar.checkbox("Show Step-by-Step", value=True, key="show_steps_gs")
    show_convergence_plot_gs = st.sidebar.checkbox("Show Convergence Plot", value=True, key="show_conv_plot_gs")
    show_dominance_check_gs = st.sidebar.checkbox("Show Diagonal Dominance Check", value=True, key="show_dom_check_gs")
    number_format_gs = st.sidebar.radio("Number Format:", ["Decimal", "Scientific"], horizontal=True, key="num_format_gs")
    
    st.sidebar.markdown("---")
    calculate_gs = st.sidebar.button("🚀 SOLVE SYSTEM", type="primary", use_container_width=True, key="calc_gs")
    
    # Main content
    if calculate_gs:
        st.session_state.has_interacted = True
        
        # Convert to numpy arrays
        A_gs = np.array(matrix_A_gs, dtype=float)
        b_gs = np.array(vector_b_gs, dtype=float)
        x0_gs = np.array(initial_guess_gs) if initial_guess_gs is not None else None
        
        # Display the system
        st.markdown("### 📐 Linear System: Ax = b")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("**Matrix A:**")
            st.code(format_matrix(A_gs), language="text")
        
        with col2:
            st.markdown("**Vector b:**")
            st.code(format_vector(b_gs), language="text")
        
        with col3:
            st.markdown("**Initial x₀:**")
            if x0_gs is not None:
                st.code(format_vector(x0_gs), language="text")
            else:
                st.code(format_vector(np.zeros(system_size)), language="text")
        
        st.markdown("---")
        
        # Display Gauss-Seidel Iteration Equations
        st.markdown("### 📐 Gauss-Seidel Iteration Equations")
        st.markdown("The Gauss-Seidel method uses **updated values immediately**:")

        # Display general formula
        st.latex(r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j<i} a_{ij} x_j^{(k+1)} - \sum_{j>i} a_{ij} x_j^{(k)} \right)")

        st.markdown("**For this system:**")

        # Create columns for better layout if system is small
        if system_size <= 3:
            cols = st.columns(system_size)
        else:
            cols = None

        for i in range(system_size):
            # Build the equation string for LaTeX
            numerator_parts = [f"{b_gs[i]:.4g}"]  # Start with b_i
            
            # Collect terms with updated values (j < i)
            updated_terms = []
            for j in range(i):
                if A_gs[i, j] != 0:
                    coeff = A_gs[i, j]
                    if coeff > 0:
                        updated_terms.append(f"- {coeff:.4g} x_{{{j+1}}}^{{(k+1)}}")
                    else:
                        updated_terms.append(f"+ {abs(coeff):.4g} x_{{{j+1}}}^{{(k+1)}}")
            
            # Collect terms with old values (j > i)
            old_terms = []
            for j in range(i + 1, system_size):
                if A_gs[i, j] != 0:
                    coeff = A_gs[i, j]
                    if coeff > 0:
                        old_terms.append(f"- {coeff:.4g} x_{{{j+1}}}^{{(k)}}")
                    else:
                        old_terms.append(f"+ {abs(coeff):.4g} x_{{{j+1}}}^{{(k)}}")
            
            # Combine all terms
            all_terms = updated_terms + old_terms
            if all_terms:
                numerator_parts.extend(all_terms)
            
            # Construct the full equation
            numerator = " ".join(numerator_parts)
            denominator = f"{A_gs[i, i]:.4g}"
            
            equation = f"x_{{{i+1}}}^{{(k+1)}} = \\frac{{{numerator}}}{{{denominator}}}"
            
            # Display in columns if system is small, otherwise stack vertically
            if cols and i < len(cols):
                with cols[i]:
                    st.latex(equation)
            else:
                st.latex(equation)

        st.info("💡 **Key Difference from Jacobi:** Gauss-Seidel uses x^(k+1) values as soon as they're computed!")
        st.markdown("---")
        
        # Check diagonal dominance first
        if show_dominance_check_gs:
            st.markdown("### 🔍 Diagonal Dominance Check")
            
            is_dominant, details = check_diagonal_dominance(A_gs)
            
            if is_dominant:
                st.success("✅ Matrix is **strictly diagonally dominant**. Gauss-Seidel method is **guaranteed to converge**!")
            else:
                st.warning("⚠️ Matrix is **NOT strictly diagonally dominant**. Convergence is **not guaranteed** but may still occur.")
            
            # Show details table
            dom_df = pd.DataFrame(details)
            st.dataframe(dom_df, use_container_width=True, hide_index=True)
            
            # Calculate spectral radius
            try:
                from methods.gauss_seidel import calculate_spectral_radius_gs
                spectral_rad = calculate_spectral_radius_gs(A_gs)
                st.markdown(f"**Spectral Radius ρ(T_GS):** {spectral_rad:.6f}")
                
                if spectral_rad < 1:
                    st.success(f"✅ ρ(T_GS) = {spectral_rad:.6f} < 1 → Method **will converge**")
                else:
                    st.error(f"❌ ρ(T_GS) = {spectral_rad:.6f} ≥ 1 → Method **may not converge**")
            except:
                st.info("ℹ️ Could not calculate spectral radius")
            
            st.markdown("---")
        
        # Solve using Gauss-Seidel
        with st.spinner("Solving system using Gauss-Seidel method..."):
            from methods.gauss_seidel import gauss_seidel_method
            result_gs = gauss_seidel_method(A_gs, b_gs, x0=x0_gs, tol=tolerance_gs, max_iter=max_iter_gs)
        
        if not result_gs['success']:
            st.error(f"❌ {result_gs['message']}")
            st.stop()
        
        # Display result
        if result_gs['converged']:
            st.success(result_gs['message'])
        else:
            st.warning(result_gs['message'])
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Converged", "✅ Yes" if result_gs['converged'] else "❌ No")
        col2.metric("Iterations", len(result_gs['iterations']) - 1)
        col3.metric("Final Error", f"{result_gs['final_error']:.2e}")
        
        # Calculate final residual
        final_residual_gs = np.linalg.norm(np.dot(A_gs, result_gs['solution']) - b_gs, ord=np.inf)
        col4.metric("Residual ||Ax-b||", f"{final_residual_gs:.2e}")
        
        st.markdown("---")
        
        # Display solution
        st.markdown("### ✅ Solution Vector x")
        
        sol_col1, sol_col2 = st.columns(2)
        
        with sol_col1:
            st.markdown("**Vector Form:**")
            st.code(format_vector(result_gs['solution']), language="text")
        
        with sol_col2:
            st.markdown("**Component Form:**")
            for i, val in enumerate(result_gs['solution']):
                if number_format_gs == "Scientific":
                    st.code(f"x{i+1} = {val:.6e}")
                else:
                    st.code(f"x{i+1} = {val:.8f}")
        
        st.markdown("---")
        
        # Verification
        st.markdown("### 🔬 Verification: Ax = b")
        
        Ax_gs = np.dot(A_gs, result_gs['solution'])
        
        ver_col1, ver_col2, ver_col3 = st.columns(3)
        
        with ver_col1:
            st.markdown("**Ax (computed):**")
            st.code(format_vector(Ax_gs), language="text")
        
        with ver_col2:
            st.markdown("**b (expected):**")
            st.code(format_vector(b_gs), language="text")
        
        with ver_col3:
            st.markdown("**Difference (Ax - b):**")
            difference_gs = Ax_gs - b_gs
            st.code(format_vector(difference_gs), language="text")
        
        st.markdown("---")
        
        # Step-by-step solution
        if show_steps_gs:
            st.markdown("### 📝 Detailed Step-by-Step Solution")
            
            # Step 1: Understanding
            with st.expander("📖 **Step 1: Understanding Gauss-Seidel Method**", expanded=True):
                st.markdown("""
                The **Gauss-Seidel method** is an improved version of Jacobi that uses updated values **immediately**.
                
                #### Algorithm:
                1. Decompose matrix A = D + L + U, where:
                   - **D** = diagonal part of A
                   - **L** = lower triangular part (below diagonal)
                   - **U** = upper triangular part (above diagonal)
                
                2. Iteration formula:
                """)
                st.latex(r"x^{(k+1)} = (D+L)^{-1}(b - Ux^{(k)})")
                
                st.markdown("Or component-wise:")
                st.latex(r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j<i} a_{ij} x_j^{(k+1)} - \sum_{j>i} a_{ij} x_j^{(k)} \right)")
                
                st.markdown("""
                #### Key Difference from Jacobi:
                - **Jacobi:** Uses all values from iteration k
                - **Gauss-Seidel:** Uses x^(k+1) values as soon as they're computed
                - **Result:** Typically **faster convergence** than Jacobi
                
                #### Convergence:
                - **Guaranteed** if A is strictly diagonally dominant or symmetric positive definite
                - **Faster** than Jacobi for most problems
                - Cannot be parallelized (sequential updates)
                """)
            
            # Step 2: Matrix decomposition
            with st.expander("🔧 **Step 2: Matrix Decomposition (A = D + L + U)**", expanded=False):
                D_gs = np.diag(np.diag(A_gs))
                L_gs = np.tril(A_gs, -1)
                U_gs = np.triu(A_gs, 1)
                
                st.markdown("**Diagonal Matrix D:**")
                st.code(format_matrix(D_gs), language="text")
                
                st.markdown("**Lower Triangular L (below diagonal):**")
                st.code(format_matrix(L_gs), language="text")
                
                st.markdown("**Upper Triangular U (above diagonal):**")
                st.code(format_matrix(U_gs), language="text")
            
            # Step 3: Iteration formula for each component
            with st.expander("📐 **Step 3: Component-wise Iteration Formulas**", expanded=False):
                st.markdown("For each variable xᵢ (computed in order):")
                
                for i in range(system_size):
                    st.markdown(f"**Variable x{i+1}:**")
                    
                    # Build formula string with updated and old values
                    updated_terms = []
                    old_terms = []
                    
                    for j in range(system_size):
                        if i != j and A_gs[i, j] != 0:
                            if j < i:
                                # Already computed in this iteration
                                updated_terms.append(f"{A_gs[i,j]:.4f}·x{j+1}^{{(k+1)}}")
                            else:
                                # Not yet computed, use old value
                                old_terms.append(f"{A_gs[i,j]:.4f}·x{j+1}^{{(k)}}")
                    
                    updated_str = " - ".join(updated_terms) if updated_terms else ""
                    old_str = " - ".join(old_terms) if old_terms else ""
                    
                    # Combine
                    if updated_str and old_str:
                        sum_str = f"({updated_str}) - ({old_str})"
                    elif updated_str:
                        sum_str = updated_str
                    elif old_str:
                        sum_str = old_str
                    else:
                        sum_str = "0"
                    
                    st.latex(f"x_{{{i+1}}}^{{(k+1)}} = \\frac{{1}}{{{A_gs[i,i]:.4f}}} \\left( {b_gs[i]:.4f} - {sum_str} \\right)")
                    
                    if i > 0:
                        st.caption(f"⚡ Uses already-updated values: x₁^(k+1) through x{i}^(k+1)")
            
            # Step 4: Show first few iterations in detail
            with st.expander("🔄 **Step 4: Iteration Details (First 3 Iterations)**", expanded=False):
                for iter_idx in range(min(4, len(result_gs['iterations']))):
                    iter_data = result_gs['iterations'][iter_idx]
                    
                    st.markdown(f"#### Iteration {iter_data['iteration']}")
                    
                    if iter_data['iteration'] == 0:
                        st.markdown("**Initial guess:**")
                        st.code(format_vector(iter_data['x']), language="text")
                    else:
                        st.markdown("**Calculations (sequential order):**")
                        
                        # Show calculation for each component
                        x_prev = result_gs['iterations'][iter_idx - 1]['x']
                        x_current = np.array(x_prev.copy())  # Will be updated component by component
                        
                        for i in range(system_size):
                            sum_val = 0.0
                            calc_parts = []
                            
                            # Use updated values for j < i
                            for j in range(i):
                                sum_val += A_gs[i, j] * x_current[j]
                                if A_gs[i, j] != 0:
                                    calc_parts.append(f"{A_gs[i,j]:.4f}×{x_current[j]:.4f}(new)")
                            
                            # Use old values for j > i
                            for j in range(i + 1, system_size):
                                sum_val += A_gs[i, j] * x_prev[j]
                                if A_gs[i, j] != 0:
                                    calc_parts.append(f"{A_gs[i,j]:.4f}×{x_prev[j]:.4f}(old)")
                            
                            sum_str = " + ".join(calc_parts) if calc_parts else "0"
                            result_val = (b_gs[i] - sum_val) / A_gs[i, i]
                            
                            st.code(f"x{i+1} = ({b_gs[i]:.4f} - ({sum_str})) / {A_gs[i,i]:.4f} = {result_val:.6f}")
                            
                            # Update current value for next iteration
                            x_current[i] = result_val
                        
                        st.markdown(f"**Updated x:**")
                        st.code(format_vector(iter_data['x']), language="text")
                    
                    st.markdown(f"**Error:** {iter_data['error']:.6e}")
                    st.markdown(f"**Residual:** {iter_data['residual']:.6e}")
                    st.markdown("---")
        
        # Iteration table
        st.markdown("### 📊 Complete Iteration History")
        
        # Prepare dataframe
        iter_display_gs = []
        for iter_data in result_gs['iterations']:
            row = {'Iteration': iter_data['iteration']}
            
            # Add components
            for i in range(system_size):
                if number_format_gs == "Scientific":
                    row[f'x{i+1}'] = f"{iter_data[f'x{i+1}']:.6e}"
                else:
                    row[f'x{i+1}'] = f"{iter_data[f'x{i+1}']:.8f}"
            
            row['Error'] = f"{iter_data['error']:.6e}"
            row['Residual'] = f"{iter_data['residual']:.6e}"
            
            iter_display_gs.append(row)
        
        iter_df_gs = pd.DataFrame(iter_display_gs)
        st.dataframe(iter_df_gs, use_container_width=True, height=400)
        
        # Download button
        csv_gs = pd.DataFrame(result_gs['iterations']).to_csv(index=False)
        st.download_button("📥 Download CSV", csv_gs, "gauss_seidel_iterations.csv", "text/csv")
        
        # Convergence plot
        if show_convergence_plot_gs and len(result_gs['iterations']) > 1:
            st.markdown("---")
            st.markdown("### 📈 Convergence Analysis")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Error plot
            iterations_gs = [it['iteration'] for it in result_gs['iterations']]
            errors_gs = [it['error'] for it in result_gs['iterations']]
            residuals_gs = [it['residual'] for it in result_gs['iterations']]
            
            ax1.semilogy(iterations_gs, errors_gs, 'g-o', linewidth=2, markersize=6, label='Error ||x^(k+1) - x^(k)||')
            ax1.axhline(y=tolerance_gs, color='red', linestyle='--', linewidth=1.5, label=f'Tolerance = {tolerance_gs:.1e}')
            ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Error (log scale)', fontsize=12, fontweight='bold')
            ax1.set_title('Convergence: Error vs Iteration', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3, which='both')
            
            # Residual plot
            ax2.semilogy(iterations_gs, residuals_gs, 'm-s', linewidth=2, markersize=6, label='Residual ||Ax - b||')
            ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Residual (log scale)', fontsize=12, fontweight='bold')
            ax2.set_title('Convergence: Residual vs Iteration', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3, which='both')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Component-wise convergence
            st.markdown("#### 📊 Component-wise Convergence")
            
            fig2, ax3 = plt.subplots(figsize=(12, 6))
            
            for i in range(system_size):
                component_vals = [it[f'x{i+1}'] for it in result_gs['iterations']]
                ax3.plot(iterations_gs, component_vals, '-o', linewidth=2, markersize=5, label=f'x{i+1}')
            
            ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Value', fontsize=12, fontweight='bold')
            ax3.set_title('Evolution of Solution Components', fontsize=14, fontweight='bold')
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
            
            st.pyplot(fig2)
    
    else:
        # Welcome screen
        st.info("👈 **Get Started:** Enter your linear system Ax = b in the sidebar and click 'SOLVE SYSTEM'")
        
        st.markdown("### 📚 About Gauss-Seidel Iterative Method")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **What is the Gauss-Seidel Method?**
            
            The Gauss-Seidel method is an **improved iterative algorithm** for solving linear systems **Ax = b**. 
            It's a refinement of the Jacobi method that typically **converges faster**.
            
            #### Algorithm:
            Starting from an initial guess x⁽⁰⁾, we iteratively update using the **latest available values**:
            """)
            st.latex(r"x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j<i} a_{ij} x_j^{(k+1)} - \sum_{j>i} a_{ij} x_j^{(k)} \right)")
            
            st.markdown("""
            #### Key Features:
            - ✅ **Faster than Jacobi** (uses latest values)
            - ✅ **Simple to implement**
            - ✅ **Lower memory usage**
            - ⚠️ **Sequential** (cannot parallelize)
            """)
        
            st.markdown("---")

# ============================================================================
# NUMERICAL DIFFERENTIATION SECTION
# ============================================================================
elif problem_type == "📈 Numerical Differentiation":
    st.sidebar.markdown("### 📈 Numerical Differentiation")
    
    # Method selection
    diff_method = st.sidebar.radio(
        "**Select Method:**",
        ["🔄 Newton Forward", "🔙 Newton Backward"],
        help="Forward: Use first point (x₀). Backward: Use last point (xₙ)"
    )
    
    # Derivative order
    calculate_all = st.sidebar.checkbox(
        "**Calculate All Derivatives**",
        value=False,
        help="Calculate 1st, 2nd, and 3rd derivatives at once"
    )
    
    if not calculate_all:
        derivative_order = st.sidebar.selectbox(
            "**Derivative Order:**",
            [1, 2, 3],
            index=0,
            help="Select which derivative to calculate"
        )
    else:
        derivative_order = None  # Will calculate all
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Points")
    
    # Number of points
    num_points = st.sidebar.number_input(
        "**Number of Points:**",
        min_value=2,
        max_value=10,
        value=7,  # Default to 7 for the provided example
        help="Enter at least 2 points (more points = higher accuracy)"
    )
    
    st.sidebar.markdown("---")
    
    # Initialize session state
    if 'diff_x_points' not in st.session_state:
        st.session_state.diff_x_points = [0.0] * 10
    if 'diff_y_points' not in st.session_state:
        st.session_state.diff_y_points = [0.0] * 10
    
    # Default example data
    default_example = {
        'x': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        'y': [7.989, 8.403, 8.781, 9.129, 9.451, 9.750, 10.031]
    }
    
    # Example data (e^x)
    example_data = {
        'x': [0.0, 0.1, 0.2, 0.3, 0.4],
        'y': [1.0, 1.1052, 1.2214, 1.3499, 1.4918]  # e^x values
    }
    
    # Load default values on first run
    if 'diff_default_loaded' not in st.session_state:
        st.session_state.diff_default_loaded = True
        for i in range(len(default_example['x'])):
            if i < len(st.session_state.diff_x_points):
                st.session_state.diff_x_points[i] = default_example['x'][i]
                st.session_state.diff_y_points[i] = default_example['y'][i]
    
    col1, col2 = st.sidebar.columns(2)
    if col1.button("📝 Load Default", use_container_width=True):
        for i in range(len(default_example['x'])):
            if i < len(st.session_state.diff_x_points):
                st.session_state.diff_x_points[i] = default_example['x'][i]
                st.session_state.diff_y_points[i] = default_example['y'][i]
        st.rerun()
    
    if col2.button("📝 Load Example (e^x)", use_container_width=True):
        for i in range(len(example_data['x'])):
            if i < len(st.session_state.diff_x_points):
                st.session_state.diff_x_points[i] = example_data['x'][i]
                st.session_state.diff_y_points[i] = example_data['y'][i]
        st.rerun()
    
    st.sidebar.markdown("### 📍 Enter Data Points")
    
    # Input points
    x_points = []
    y_points = []
    
    for i in range(num_points):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            # Calculate default x value from previous point or use default example
            if i == 0:
                default_x = float(st.session_state.diff_x_points[i]) if i < len(st.session_state.diff_x_points) else (float(default_example['x'][i]) if i < len(default_example['x']) else 0.0)
            else:
                # Use previous x value + estimated h if available, or use default example
                prev_x = float(st.session_state.diff_x_points[i-1]) if (i-1) < len(st.session_state.diff_x_points) else (float(default_example['x'][i-1]) if (i-1) < len(default_example['x']) else 0.0)
                if i > 1:
                    prev_prev_x = float(st.session_state.diff_x_points[i-2]) if (i-2) >= 0 and (i-2) < len(st.session_state.diff_x_points) else (float(default_example['x'][i-2]) if (i-2) >= 0 and (i-2) < len(default_example['x']) else 0.0)
                    estimated_h = prev_x - prev_prev_x if prev_prev_x != 0 else 0.1
                else:
                    estimated_h = 0.1
                default_x = float(st.session_state.diff_x_points[i]) if i < len(st.session_state.diff_x_points) else (float(default_example['x'][i]) if i < len(default_example['x']) else prev_x + estimated_h)
            
            x_val = col1.number_input(
                f"x{i}",
                value=default_x,
                format="%.6f",
                key=f"diff_x_{i}",
                help=f"x-coordinate for point {i}"
            )
        with col2:
            y_val = col2.number_input(
                f"y{i}",
                value=float(st.session_state.diff_y_points[i]) if i < len(st.session_state.diff_y_points) else 0.0,
                format="%.6f",
                key=f"diff_y_{i}"
            )
        x_points.append(x_val)
        y_points.append(y_val)
        st.session_state.diff_x_points[i] = x_val
        st.session_state.diff_y_points[i] = y_val
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Point of Differentiation")
    
    # Show available x values
    if len(x_points) > 0:
        st.sidebar.info(f"📊 Available x values: {min(x_points):.4f} to {max(x_points):.4f}")
        
        # Let user select or input x₀
        selection_method = st.sidebar.radio(
            "**How to select point?**",
            ["Select from dropdown", "Enter manually"],
            help="Choose how to specify the point at which to calculate the derivative"
        )
        
        if selection_method == "Select from dropdown":
            # Default index based on method
            default_idx = 0 if diff_method == "🔄 Newton Forward" else len(x_points) - 1
            
            x0_value = st.sidebar.selectbox(
                f"**Select x{'₀' if diff_method == '🔄 Newton Forward' else 'ₙ'} (point to differentiate at):**",
                options=x_points,
                index=default_idx,
                help="Forward: Choose point near start. Backward: Choose point near end."
            )
            x0_index = x_points.index(x0_value)
        else:
            # Manual input
            default_x0 = float(x_points[0]) if diff_method == "🔄 Newton Forward" else float(x_points[-1])
            x0_value = st.sidebar.number_input(
                f"**Enter x{'₀' if diff_method == '🔄 Newton Forward' else 'ₙ'}:",
                value=default_x0,
                format="%.6f",
                help="Enter the x value at which to calculate the derivative"
            )
            
            # Validate if x0_value exists in dataset
            if x0_value not in x_points:
                st.sidebar.error(f"❌ x = {x0_value:.6f} not in dataset!")
                st.sidebar.warning(f"💡 Available values: {', '.join([f'{x:.6f}' for x in x_points])}")
                x0_index = None
            else:
                x0_index = x_points.index(x0_value)
        
        if x0_index is not None:
            st.sidebar.success(f"✅ Using x = {x0_value:.6f} (index {x0_index})")
            
            # Warning for forward method if x0 is near the end
            if diff_method == "🔄 Newton Forward" and x0_index > len(x_points) // 2:
                st.sidebar.warning(f"⚠️ Forward method works best near the start. You have {len(x_points) - x0_index} points after x₀.")
            
            # Warning for backward method if xn is near the start
            if diff_method == "🔙 Newton Backward" and x0_index < len(x_points) // 2:
                st.sidebar.warning(f"⚠️ Backward method works best near the end. You have {x0_index + 1} points before xₙ.")
    else:
        x0_index = None
        x0_value = None
        st.sidebar.warning("⚠️ Enter data points first")
    
    # Calculate button
    calculate_diff = st.sidebar.button("🚀 CALCULATE DERIVATIVE", type="primary", use_container_width=True)
    
    # Main content
    st.markdown("## 📈 Numerical Differentiation")
    st.markdown(f"### Using **{diff_method}** Difference Formula")
    
    if calculate_diff:
        try:
            # Validate data
            if len(x_points) != len(y_points):
                st.error("❌ Number of x and y values must match!")
                st.stop()
            
            if len(x_points) < 2:
                st.error("❌ Need at least 2 data points!")
                st.stop()
            
            # Validate x0_index
            if x0_index is None:
                st.error("❌ Please select a valid point to differentiate at!")
                st.stop()
            
            # Validate equally spaced points
            from methods.newton_differentiation import validate_equally_spaced
            is_valid, calculated_h = validate_equally_spaced(x_points)
            if not is_valid:
                st.error("❌ Points must be equally spaced! Please ensure constant step size h between consecutive x values.")
                st.warning(f"💡 Detected step size: h = {calculated_h:.6f}, but points are not uniformly spaced.")
                st.stop()
            
            # Use calculated h instead of user input
            h = calculated_h
            
            # Calculate derivative(s)
            if calculate_all:
                # Calculate all three derivatives
                results = {}
                details_dict = {}
                
                # Check minimum points
                if len(x_points) < 2:
                    st.error("❌ Need at least 2 points for 1st derivative!")
                    st.stop()
                if len(x_points) < 3:
                    st.warning("⚠️ Need at least 3 points for 2nd derivative. Calculating only 1st derivative.")
                if len(x_points) < 4:
                    st.warning("⚠️ Need at least 4 points for 3rd derivative. Calculating only 1st and 2nd derivatives.")
                
                # Calculate 1st derivative
                try:
                    if diff_method == "🔄 Newton Forward":
                        result_dict = newton_forward_derivative(x_points, y_points, 1, x0_index=x0_index, max_terms=None)
                    else:
                        result_dict = newton_backward_derivative(x_points, y_points, 1, xn_index=x0_index, max_terms=None)
                    
                    if result_dict['success']:
                        results[1] = result_dict['derivative']
                        details_dict[1] = result_dict
                    else:
                        st.error(f"❌ Error calculating 1st derivative: {result_dict['message']}")
                        results[1] = None
                except Exception as e:
                    st.error(f"❌ Error calculating 1st derivative: {str(e)}")
                    results[1] = None
                
                # Calculate 2nd derivative
                if len(x_points) >= 3:
                    try:
                        if diff_method == "🔄 Newton Forward":
                            result_dict = newton_forward_derivative(x_points, y_points, 2, x0_index=x0_index, max_terms=None)
                        else:
                            result_dict = newton_backward_derivative(x_points, y_points, 2, xn_index=x0_index, max_terms=None)
                        
                        if result_dict['success']:
                            results[2] = result_dict['derivative']
                            details_dict[2] = result_dict
                        else:
                            st.warning(f"⚠️ Could not calculate 2nd derivative: {result_dict['message']}")
                            results[2] = None
                    except Exception as e:
                        st.warning(f"⚠️ Could not calculate 2nd derivative: {str(e)}")
                        results[2] = None
                
                # Calculate 3rd derivative
                if len(x_points) >= 4:
                    try:
                        if diff_method == "🔄 Newton Forward":
                            result_dict = newton_forward_derivative(x_points, y_points, 3, x0_index=x0_index, max_terms=None)
                        else:
                            result_dict = newton_backward_derivative(x_points, y_points, 3, xn_index=x0_index, max_terms=None)
                        
                        if result_dict['success']:
                            results[3] = result_dict['derivative']
                            details_dict[3] = result_dict
                        else:
                            st.warning(f"⚠️ Could not calculate 3rd derivative: {result_dict['message']}")
                            results[3] = None
                    except Exception as e:
                        st.warning(f"⚠️ Could not calculate 3rd derivative: {str(e)}")
                        results[3] = None
                
                # Set result and details for compatibility
                result = None
                details = None
            else:
                # Check minimum points for derivative order
                min_points = derivative_order + 1
                if len(x_points) < min_points:
                    st.error(f"❌ Need at least {min_points} points for {derivative_order}{'st' if derivative_order == 1 else 'nd' if derivative_order == 2 else 'rd'} derivative!")
                    st.stop()
                
                # Calculate single derivative using generalized function
                if diff_method == "🔄 Newton Forward":
                    result_dict = newton_forward_derivative(x_points, y_points, derivative_order, x0_index=x0_index, max_terms=None)
                else:  # Backward
                    result_dict = newton_backward_derivative(x_points, y_points, derivative_order, xn_index=x0_index, max_terms=None)
                
                if not result_dict['success']:
                    st.error(f"❌ {result_dict['message']}")
                    st.stop()
                
                result = result_dict['derivative']
                details = result_dict
            
            # Display results
            st.markdown("---")
            st.markdown("### ✅ Results")
            
            # Show validation success - h is calculated from x points
            calculated_h = details.get('spacing_h', 0.1) if not calculate_all else (details_dict.get(1, {}).get('spacing_h', 0.1) if results.get(1) is not None else 0.1)
            st.success(f"✅ Points are equally spaced with h = {calculated_h:.8f}")
            
            if calculate_all:
                # Display all derivatives
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if results.get(1) is not None:
                        st.metric("1st Derivative", f"{results[1]:.8f}")
                    else:
                        st.metric("1st Derivative", "N/A")
                with col2:
                    if results.get(2) is not None:
                        st.metric("2nd Derivative", f"{results[2]:.8f}")
                    else:
                        st.metric("2nd Derivative", "N/A")
                with col3:
                    if results.get(3) is not None:
                        st.metric("3rd Derivative", f"{results[3]:.8f}")
                    else:
                        st.metric("3rd Derivative", "N/A")
                with col4:
                    calc_h_display = details_dict.get(1, {}).get('spacing_h', 0.1) if results.get(1) is not None else 0.1
                    st.metric("Step Size (h)", f"{calc_h_display:.8f}")
                
                st.metric("Data Points Used", len(x_points))
                
                # Point of evaluation
                eval_point = x_points[0] if diff_method == "🔄 Newton Forward" else x_points[-1]
                eval_label = "x₀" if diff_method == "🔄 Newton Forward" else "xₙ"
                
                st.info(f"💡 Derivatives calculated at **{eval_label} = {eval_point:.6f}**")
                
                st.markdown("---")
                st.markdown("### 📐 Formulas Used")
                
                # Show all formulas
                for order in [1, 2, 3]:
                    if results.get(order) is not None and details_dict.get(order) is not None:
                        st.markdown(f"#### {order}{'st' if order == 1 else 'nd' if order == 2 else 'rd'} Derivative:")
                        formula = details_dict[order].get('formula_used', details_dict[order].get('formula', ''))
                        if formula:
                            st.latex(formula)
            else:
                # Display single derivative
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        f"{derivative_order}{'st' if derivative_order == 1 else 'nd' if derivative_order == 2 else 'rd'} Derivative",
                        f"{result:.8f}"
                    )
                with col2:
                    st.metric("Step Size (h)", f"{calculated_h:.8f}")
                with col3:
                    st.metric("Data Points Used", len(x_points))
                
                # Point of evaluation
                eval_point = details.get('at_point', x_points[0] if diff_method == "🔄 Newton Forward" else x_points[-1])
                eval_label = "x₀" if diff_method == "🔄 Newton Forward" else "xₙ"
                
                st.info(f"💡 Derivative calculated at **{eval_label} = {eval_point:.6f}**")
                
                st.markdown("---")
                st.markdown("### 📐 Formula Used")
                formula = details.get('formula_used', details.get('formula', ''))
                if formula:
                    st.latex(formula)
            
            # Detailed calculation
            st.markdown("---")
            st.markdown("### 🔍 Detailed Calculation")
            
            with st.expander("📊 **Difference Table**", expanded=True):
                # Get difference table from result
                if calculate_all:
                    # Use the first available result's difference table
                    diff_table_2d = None
                    for order in [1, 2, 3]:
                        if results.get(order) is not None and details_dict.get(order) is not None:
                            diff_table_2d = details_dict[order].get('difference_table')
                            break
                else:
                    diff_table_2d = details.get('difference_table')
                
                if diff_table_2d is None:
                    # Fallback: calculate directly
                    from methods.newton_differentiation import calculate_forward_differences, calculate_backward_differences
                    if diff_method == "🔄 Newton Forward":
                        diff_table_2d = calculate_forward_differences(y_points, len(y_points) - 1)
                    else:
                        diff_table_2d = calculate_backward_differences(y_points, len(y_points) - 1)
                
                if diff_method == "🔄 Newton Forward":
                    st.markdown("#### Forward Differences:")
                else:
                    st.markdown("#### Backward Differences:")
                
                # Convert 2D difference table to display format
                if diff_table_2d and len(diff_table_2d) > 0:
                    max_order = len(diff_table_2d[0]) - 1
                    table_data = []
                    
                    # Determine which values are used in the formula
                    used_cells = {}  # {(row_index, column_name): True}
                    
                    if calculate_all:
                        # For all derivatives, mark all used indices from results
                        for order in [1, 2, 3]:
                            if results.get(order) is not None and details_dict.get(order) is not None:
                                det = details_dict[order]
                                terms_used = det.get('terms_used', [])
                                x0_idx = det.get('x0_index', 0) if diff_method == "🔄 Newton Forward" else det.get('xn_index', len(y_points) - 1)
                                
                                for term in terms_used:
                                    k = term.get('order', 0)
                                    if k > 0:
                                        diff_symbol = ("Δ" * k) if diff_method == "🔄 Newton Forward" else ("∇" * k)
                                        used_cells[(x0_idx, f'{diff_symbol}y')] = True
                    else:
                        # For single derivative - use ALL terms from result
                        terms_used = details.get('terms_used', [])
                        x0_idx = details.get('x0_index', 0) if diff_method == "🔄 Newton Forward" else details.get('xn_index', len(y_points) - 1)
                        
                        for term in terms_used:
                            k = term.get('order', 0)
                            if k > 0:
                                diff_symbol = ("Δ" * k) if diff_method == "🔄 Newton Forward" else ("∇" * k)
                                used_cells[(x0_idx, f'{diff_symbol}y')] = True
                    
                    # Build table data from 2D difference table
                    for i in range(len(y_points)):
                        row = {'i': i, 'xᵢ': f"{x_points[i]:.6f}", 'yᵢ': f"{y_points[i]:.6f}"}
                        
                        for order in range(1, max_order + 1):
                            if i < len(diff_table_2d) and order < len(diff_table_2d[i]):
                                if diff_method == "🔄 Newton Forward":
                                    # Forward: show differences starting from index 0
                                    if i < len(diff_table_2d) - order + 1:
                                        diff_symbol = "Δ" * order
                                        row[f'{diff_symbol}y'] = diff_table_2d[i][order]
                                else:
                                    # Backward: show differences ending at index
                                    if i >= order:
                                        diff_symbol = "∇" * order
                                        row[f'{diff_symbol}y'] = diff_table_2d[i][order]
                        
                        table_data.append(row)
                    
                    # Create DataFrame
                    df = pd.DataFrame(table_data)
                    
                    # Apply styling to highlight used cells
                    def highlight_used_cells(row):
                        styles = [''] * len(row)
                        for col_idx, col_name in enumerate(df.columns):
                            if (row.name, col_name) in used_cells:
                                styles[col_idx] = 'background-color: #ffd700; font-weight: bold;'
                        return styles
                    
                    # Style the dataframe
                    styled_df = df.style.apply(highlight_used_cells, axis=1)
                    
                    # Format numeric columns (difference columns)
                    diff_columns = [col for col in df.columns if col not in ['i', 'xᵢ', 'yᵢ']]
                    if diff_columns:
                        styled_df = styled_df.format("{:.8f}", subset=diff_columns)
                    
                    st.markdown("**⭐ Highlighted cells (gold) = Values used in formula**")
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    
                    # Show maximum order reached
                    if max_order > 0:
                        st.caption(f"📊 Maximum difference order calculated: **{max_order}** (until one value remains)")
                else:
                    st.warning("⚠️ Could not generate difference table")
            
            # Step-by-step calculation
            if calculate_all:
                # Show step-by-step for all calculated derivatives
                for order in [1, 2, 3]:
                    if results.get(order) is not None and details_dict.get(order) is not None:
                        det = details_dict[order]
                        res = results[order]
                        with st.expander(f"🧮 **Step-by-Step: {order}{'st' if order == 1 else 'nd' if order == 2 else 'rd'} Derivative**", expanded=(order == 1)):
                            st.markdown(f"#### Calculating {order}{'st' if order == 1 else 'nd' if order == 2 else 'rd'} derivative using {det.get('method', 'Newton')} formula:")
                            formula = det.get('formula_used', det.get('formula', ''))
                            if formula:
                                st.latex(formula)
                            st.markdown("---")
                            
                            # Get difference table and terms used
                            diff_table = det.get('difference_table', [])
                            terms_used = det.get('terms_used', [])
                            calc_h = det.get('spacing_h', 0.1)
                            
                            if diff_table and len(diff_table) > 0 and terms_used:
                                st.markdown(f"**Step 1:** Extract differences from difference table")
                                st.markdown(f"**Total terms used:** {len(terms_used)}")
                                
                                x0_idx = det.get('x0_index', 0) if diff_method == "🔄 Newton Forward" else det.get('xn_index', len(y_points) - 1)
                                
                                if diff_method == "🔄 Newton Forward":
                                    # Forward: extract from row x0_index
                                    for term in terms_used:  # Show ALL terms
                                        k = term.get('order', 0)
                                        diff_val = term.get('difference', 0)
                                        if k > 0:
                                            diff_symbol = "Δ" * k
                                            st.latex(f"\\{diff_symbol} y_{{{x0_idx}}} = {diff_val:.8f}")
                                else:
                                    # Backward: extract from row xn_index
                                    for term in terms_used:  # Show ALL terms
                                        k = term.get('order', 0)
                                        diff_val = term.get('difference', 0)
                                        if k > 0:
                                            diff_symbol = "∇" * k
                                            st.latex(f"\\{diff_symbol} y_{{{x0_idx}}} = {diff_val:.8f}")
                                
                                st.markdown("---")
                                st.markdown("**Step 2:** Apply the formula with coefficients")
                                
                                # Build formula string
                                formula_parts = []
                                for term in terms_used:
                                    k = term.get('order', 0)
                                    coeff = term.get('coefficient', 0)
                                    diff_val = term.get('difference', 0)
                                    contrib = term.get('contribution', 0)
                                    
                                    if diff_method == "🔄 Newton Forward":
                                        diff_symbol = "Δ" * k
                                        if k == 1:
                                            formula_parts.append(f"{coeff:.6f} \\times {diff_val:.8f}")
                                        else:
                                            formula_parts.append(f"{coeff:.6f} \\times {diff_val:.8f}")
                                    else:
                                        diff_symbol = "∇" * k
                                        formula_parts.append(f"{coeff:.6f} \\times {diff_val:.8f}")
                                
                                # Show ALL terms, not just first few
                                if order == 1:
                                    if diff_method == "🔄 Newton Forward":
                                        term_parts = []
                                        for i, t in enumerate(terms_used):
                                            k = t.get('order', 1)
                                            diff_val = t.get('difference', 0)
                                            sign = "-" if i > 0 and i % 2 == 0 else "+" if i > 0 else ""
                                            if k == 1:
                                                term_parts.append(f"{sign} {diff_val:.8f}" if sign else f"{diff_val:.8f}")
                                            else:
                                                term_parts.append(f"{sign} \\frac{{{diff_val:.8f}}}{{{k}}}" if sign else f"\\frac{{{diff_val:.8f}}}{{{k}}}")
                                        terms_str = " ".join(term_parts)
                                        st.latex(f"\\frac{{dy}}{{dx}} = \\frac{{1}}{{{calc_h:.8f}}} \\left( {terms_str} \\right)")
                                    else:
                                        term_parts = []
                                        for i, t in enumerate(terms_used):
                                            k = t.get('order', 1)
                                            diff_val = t.get('difference', 0)
                                            sign = " + " if i > 0 else ""
                                            term_parts.append(f"{sign}\\frac{{{diff_val:.8f}}}{{{k}}}")
                                        terms_str = "".join(term_parts)
                                        st.latex(f"\\frac{{dy}}{{dx}} = \\frac{{1}}{{{calc_h:.8f}}} \\left( {terms_str} \\right)")
                                elif order == 2:
                                    term_parts = []
                                    for i, t in enumerate(terms_used):
                                        coeff = t.get('coefficient', 0)
                                        diff_val = t.get('difference', 0)
                                        sign = " + " if i > 0 else ""
                                        term_parts.append(f"{sign}{coeff:.6f} \\times {diff_val:.8f}")
                                    terms_str = "".join(term_parts)
                                    st.latex(f"\\frac{{d^2y}}{{dx^2}} = \\frac{{1}}{{{calc_h:.8f}^2}} \\left( {terms_str} \\right)")
                                else:
                                    term_parts = []
                                    for i, t in enumerate(terms_used):
                                        coeff = t.get('coefficient', 0)
                                        diff_val = t.get('difference', 0)
                                        sign = " + " if i > 0 else ""
                                        term_parts.append(f"{sign}{coeff:.6f} \\times {diff_val:.8f}")
                                    terms_str = "".join(term_parts)
                                    st.latex(f"\\frac{{d^3y}}{{dx^3}} = \\frac{{1}}{{{calc_h:.8f}^3}} \\left( {terms_str} \\right)")
                                
                                st.markdown("---")
                                st.markdown("**Step 3:** Final Result")
                                st.latex(f"\\frac{{d^{order}y}}{{dx^{order}}} = {res:.8f}")
            else:
                # Single derivative step-by-step
                with st.expander("🧮 **Step-by-Step Calculation**", expanded=True):
                    method_name = details.get('method', 'Newton Forward' if diff_method == "🔄 Newton Forward" else 'Newton Backward')
                    st.markdown(f"#### Calculating {derivative_order}{'st' if derivative_order == 1 else 'nd' if derivative_order == 2 else 'rd'} derivative using {method_name} formula:")
                    formula = details.get('formula_used', details.get('formula', ''))
                    if formula:
                        st.latex(formula)
                    st.markdown("---")
                    
                    # Get difference table and terms used
                    diff_table = details.get('difference_table', [])
                    terms_used = details.get('terms_used', [])
                    calc_h = details.get('spacing_h', 0.1)
                    
                    if diff_table and len(diff_table) > 0 and terms_used:
                        st.markdown(f"**Step 1:** Extract differences from difference table")
                        st.markdown(f"**Total terms used:** {len(terms_used)}")
                        
                        x0_idx = details.get('x0_index', 0) if diff_method == "🔄 Newton Forward" else details.get('xn_index', len(y_points) - 1)
                        
                        if diff_method == "🔄 Newton Forward":
                            # Forward: extract from row x0_index
                            for term in terms_used:  # Show ALL terms
                                k = term.get('order', 0)
                                diff_val = term.get('difference', 0)
                                if k > 0:
                                    diff_symbol = "Δ" * k
                                    st.latex(f"\\{diff_symbol} y_{{{x0_idx}}} = {diff_val:.8f}")
                        else:
                            # Backward: extract from row xn_index
                            for term in terms_used:  # Show ALL terms
                                k = term.get('order', 0)
                                diff_val = term.get('difference', 0)
                                if k > 0:
                                    diff_symbol = "∇" * k
                                    st.latex(f"\\{diff_symbol} y_{{{x0_idx}}} = {diff_val:.8f}")
                        
                        st.markdown("---")
                        st.markdown("**Step 2:** Apply the formula with coefficients")
                        
                        # Build weighted sum display
                        weighted_sum = sum(term.get('contribution', 0) for term in terms_used)
                        
                        if derivative_order == 1:
                            if diff_method == "🔄 Newton Forward":
                                term_parts = []
                                for i, t in enumerate(terms_used):
                                    k = t.get('order', 1)
                                    diff_val = t.get('difference', 0)
                                    sign = "-" if i > 0 and i % 2 == 0 else "+" if i > 0 else ""
                                    if k == 1:
                                        term_parts.append(f"{sign} {diff_val:.8f}" if sign else f"{diff_val:.8f}")
                                    else:
                                        term_parts.append(f"{sign} \\frac{{{diff_val:.8f}}}{{{k}}}" if sign else f"\\frac{{{diff_val:.8f}}}{{{k}}}")
                                terms_str = " ".join(term_parts)
                                st.latex(f"\\frac{{dy}}{{dx}} = \\frac{{1}}{{{calc_h:.8f}}} \\left( {terms_str} \\right)")
                            else:
                                term_parts = []
                                for i, t in enumerate(terms_used):
                                    k = t.get('order', 1)
                                    diff_val = t.get('difference', 0)
                                    sign = " + " if i > 0 else ""
                                    term_parts.append(f"{sign}\\frac{{{diff_val:.8f}}}{{{k}}}")
                                terms_str = "".join(term_parts)
                                st.latex(f"\\frac{{dy}}{{dx}} = \\frac{{1}}{{{calc_h:.8f}}} \\left( {terms_str} \\right)")
                        elif derivative_order == 2:
                            term_parts = []
                            for i, t in enumerate(terms_used):
                                coeff = t.get('coefficient', 0)
                                diff_val = t.get('difference', 0)
                                sign = " + " if i > 0 else ""
                                term_parts.append(f"{sign}{coeff:.6f} \\times {diff_val:.8f}")
                            terms_str = "".join(term_parts)
                            st.latex(f"\\frac{{d^2y}}{{dx^2}} = \\frac{{1}}{{{calc_h:.8f}^2}} \\left( {terms_str} \\right)")
                        else:
                            term_parts = []
                            for i, t in enumerate(terms_used):
                                coeff = t.get('coefficient', 0)
                                diff_val = t.get('difference', 0)
                                sign = " + " if i > 0 else ""
                                term_parts.append(f"{sign}{coeff:.6f} \\times {diff_val:.8f}")
                            terms_str = "".join(term_parts)
                            st.latex(f"\\frac{{d^3y}}{{dx^3}} = \\frac{{1}}{{{calc_h:.8f}^3}} \\left( {terms_str} \\right)")
                        
                        st.latex(f"\\frac{{d^{derivative_order}y}}{{dx^{derivative_order}}} = {result:.8f}")
                    else:
                        st.warning("⚠️ Difference table not available for step-by-step display")
            
            # Visualization
            st.markdown("---")
            st.markdown("### 📊 Data Visualization")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_points, y_points, 'bo-', linewidth=2, markersize=8, label='Data Points')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y = f(x)', fontsize=12)
            ax.set_title(f'Data Points for Numerical Differentiation', fontsize=14, fontweight='bold')
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ Error calculating derivative: {str(e)}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
    
    else:
        # Welcome screen
        st.info("👈 **Get Started:** Enter your data points in the sidebar and click 'CALCULATE DERIVATIVE'")
        
        st.markdown("### 📚 About Numerical Differentiation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **What is Numerical Differentiation?**
            
            Numerical differentiation is used to approximate derivatives when:
            - The function is only known at discrete points
            - The function is too complex to differentiate analytically
            - We have experimental data
            
            #### Newton's Forward Difference Formula:
            Used when we want the derivative at the **first point** (x₀):
            - First Derivative: Uses Δy₀, Δ²y₀, Δ³y₀, ...
            - Best for: Beginning of data range
            """)
        
        with col2:
            st.markdown("""
            **Newton's Backward Difference Formula:**
            Used when we want the derivative at the **last point** (xₙ):
            - First Derivative: Uses ∇yₙ, ∇²yₙ, ∇³yₙ, ...
            - Best for: End of data range
            
            #### Key Features:
            - ✅ Works with tabulated data
            - ✅ No need for analytical function
            - ✅ Higher order differences = better accuracy
            - ⚠️ Requires uniform spacing (h)
            """)
        
        st.markdown("---")
        st.markdown("### 📐 Formulas Reference")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔄 Newton Forward:")
            st.latex(r"\left(\frac{dy}{dx}\right)_{x=x_0} = \frac{1}{h} \left[\Delta y_0 - \frac{\Delta^2 y_0}{2} + \frac{\Delta^3 y_0}{3} - \frac{\Delta^4 y_0}{4} + \dots \right]")
            st.latex(r"\left(\frac{d^2y}{dx^2}\right)_{x=x_0} = \frac{1}{h^2} \left[\Delta^2 y_0 - \Delta^3 y_0 + \frac{11}{12} \Delta^4 y_0 + \dots \right]")
            st.latex(r"\left(\frac{d^3y}{dx^3}\right)_{x=x_0} = \frac{1}{h^3} \left[\Delta^3 y_0 - \frac{3}{2} \Delta^4 y_0 + \dots \right]")
        
        with col2:
            st.markdown("#### 🔙 Newton Backward:")
            st.latex(r"\left(\frac{dy}{dx}\right)_{x=x_n} = \frac{1}{h} \left[\nabla y_n + \frac{\nabla^2 y_n}{2} + \frac{\nabla^3 y_n}{3} + \frac{\nabla^4 y_n}{4} + \dots \right]")
            st.latex(r"\left(\frac{d^2y}{dx^2}\right)_{x=x_n} = \frac{1}{h^2} \left[\nabla^2 y_n + \nabla^3 y_n + \frac{11}{12} \nabla^4 y_n + \dots \right]")
            st.latex(r"\left(\frac{d^3y}{dx^3}\right)_{x=x_n} = \frac{1}{h^3} \left[\nabla^3 y_n + \frac{3}{2} \nabla^4 y_n + \dots \right]")

# ============================================================================
# NUMERICAL INTEGRATION SECTION
# ============================================================================
elif problem_type == "📐 Numerical Integration":
    st.sidebar.markdown("### 📐 Numerical Integration")
    
    # Method selection
    integration_method = st.sidebar.selectbox(
        "**Select Integration Method:**",
        [
            "Trapezoidal Rule",
            "Simpson's 1/3 Rule",
            "Simpson's 3/8 Rule"
        ],
        help="Choose numerical integration method"
    )
    
    # Method info
    method_info = {
        "Trapezoidal Rule": {
            'description': '🟢 Simple | Works with any number of points',
            'requirement': 'Minimum 2 points',
            'accuracy': 'O(h²)',
            'key': 'trapezoidal'
        },
        "Simpson's 1/3 Rule": {
            'description': '🟡 More Accurate | Requires ODD number of points',
            'requirement': 'Points: 3, 5, 7, 9, 11, ...',
            'accuracy': 'O(h⁴)',
            'key': 'simpson_1_3'
        },
        "Simpson's 3/8 Rule": {
            'description': '🔴 Most Accurate | Requires specific point count',
            'requirement': 'Points: 4, 7, 10, 13, 16, ...',
            'accuracy': 'O(h⁴)',
            'key': 'simpson_3_8'
        }
    }
    
    info = method_info[integration_method]
    st.sidebar.info(f"{info['description']}\n\n**Requirement:** {info['requirement']}\n**Accuracy:** {info['accuracy']}")
    
    st.sidebar.markdown("---")
    
    # Number of points
    num_points = st.sidebar.number_input(
        "**Number of Data Points:**",
        min_value=2,
        max_value=100,
        value=5,
        step=1,
        help="Enter the number of (x, y) data points"
    )
    
    # Validate point count for selected method
    method_key = info['key']
    if method_key == 'simpson_1_3':
        if num_points % 2 == 0:
            st.sidebar.error(f"❌ Simpson's 1/3 needs ODD number of points. You have {num_points}.")
    elif method_key == 'simpson_3_8':
        n = num_points - 1
        if n % 3 != 0:
            st.sidebar.error(f"❌ Simpson's 3/8 needs intervals divisible by 3. You have {n} intervals (need 3, 6, 9, ...).")
    
    st.sidebar.markdown("---")
    
    # Data input method selection
    input_method = st.sidebar.radio(
        "**Data Input Method:**",
        ["Manual Entry", "Function Input"],
        help="Choose how to provide data: manually enter points or input a function"
    )
    
    # Initialize session state
    if 'int_x_points' not in st.session_state:
        st.session_state.int_x_points = [0.0] * 100
    if 'int_y_points' not in st.session_state:
        st.session_state.int_y_points = [0.0] * 100
    if 'int_func_input' not in st.session_state:
        st.session_state.int_func_input = "x**2"
    if 'int_a' not in st.session_state:
        st.session_state.int_a = 0.0
    if 'int_b' not in st.session_state:
        st.session_state.int_b = 4.0
    if 'int_n' not in st.session_state:
        st.session_state.int_n = 4
    
    if input_method == "Function Input":
        st.sidebar.markdown("### 🔢 Function Input")
        st.sidebar.info("💡 Enter a function f(x) and integration limits. Points will be generated automatically.")
        
        # Function input
        func_str = st.sidebar.text_area(
            "**f(x) =**",
            value=st.session_state.int_func_input,
            height=80,
            help="Enter your function using the keypad below",
            key="int_function_input"
        )
        st.session_state.int_func_input = func_str
        
        # Virtual Keypad
        st.sidebar.markdown("#### 🧮 Keypad")
        
        # Powers & Roots
        col1, col2, col3, col4 = st.sidebar.columns(4)
        if col1.button("x²", use_container_width=True, key="int_pow2"):
            st.session_state.int_func_input += "x**2"
            st.rerun()
        if col2.button("x³", use_container_width=True, key="int_pow3"):
            st.session_state.int_func_input += "x**3"
            st.rerun()
        if col3.button("xⁿ", use_container_width=True, key="int_pown"):
            st.session_state.int_func_input += "x**"
            st.rerun()
        if col4.button("√x", use_container_width=True, key="int_sqrt"):
            st.session_state.int_func_input += "sqrt(x)"
            st.rerun()
        
        # Operators
        col1, col2, col3, col4 = st.sidebar.columns(4)
        if col1.button("+", use_container_width=True, key="int_add"):
            st.session_state.int_func_input += " + "
            st.rerun()
        if col2.button("−", use_container_width=True, key="int_sub"):
            st.session_state.int_func_input += " - "
            st.rerun()
        if col3.button("×", use_container_width=True, key="int_mul"):
            st.session_state.int_func_input += "*"
            st.rerun()
        if col4.button("÷", use_container_width=True, key="int_div"):
            st.session_state.int_func_input += "/"
            st.rerun()
        
        # Trig functions
        col1, col2, col3, col4 = st.sidebar.columns(4)
        if col1.button("sin", use_container_width=True, key="int_sin"):
            st.session_state.int_func_input += "sin(x)"
            st.rerun()
        if col2.button("cos", use_container_width=True, key="int_cos"):
            st.session_state.int_func_input += "cos(x)"
            st.rerun()
        if col3.button("tan", use_container_width=True, key="int_tan"):
            st.session_state.int_func_input += "tan(x)"
            st.rerun()
        if col4.button("π", use_container_width=True, key="int_pi"):
            st.session_state.int_func_input += "pi"
            st.rerun()
        
        # Advanced functions
        st.sidebar.markdown("##### Advanced")
        col1, col2, col3, col4 = st.sidebar.columns(4)
        if col1.button("eˣ", use_container_width=True, key="int_exp"):
            st.session_state.int_func_input += "exp(x)"
            st.rerun()
        if col2.button("ln", use_container_width=True, key="int_log"):
            st.session_state.int_func_input += "log(x)"
            st.rerun()
        if col3.button("|x|", use_container_width=True, key="int_abs"):
            st.session_state.int_func_input += "abs(x)"
            st.rerun()
        if col4.button("x", use_container_width=True, key="int_x"):
            st.session_state.int_func_input += "x"
            st.rerun()
        
        # Numbers
        st.sidebar.markdown("##### Numbers")
        col1, col2, col3, col4 = st.sidebar.columns(4)
        if col1.button("7", use_container_width=True, key="int_7"):
            st.session_state.int_func_input += "7"
            st.rerun()
        if col2.button("8", use_container_width=True, key="int_8"):
            st.session_state.int_func_input += "8"
            st.rerun()
        if col3.button("9", use_container_width=True, key="int_9"):
            st.session_state.int_func_input += "9"
            st.rerun()
        if col4.button("(", use_container_width=True, key="int_lpar"):
            st.session_state.int_func_input += "("
            st.rerun()
        
        col1, col2, col3, col4 = st.sidebar.columns(4)
        if col1.button("4", use_container_width=True, key="int_4"):
            st.session_state.int_func_input += "4"
            st.rerun()
        if col2.button("5", use_container_width=True, key="int_5"):
            st.session_state.int_func_input += "5"
            st.rerun()
        if col3.button("6", use_container_width=True, key="int_6"):
            st.session_state.int_func_input += "6"
            st.rerun()
        if col4.button(")", use_container_width=True, key="int_rpar"):
            st.session_state.int_func_input += ")"
            st.rerun()
        
        col1, col2, col3, col4 = st.sidebar.columns(4)
        if col1.button("1", use_container_width=True, key="int_1"):
            st.session_state.int_func_input += "1"
            st.rerun()
        if col2.button("2", use_container_width=True, key="int_2"):
            st.session_state.int_func_input += "2"
            st.rerun()
        if col3.button("3", use_container_width=True, key="int_3"):
            st.session_state.int_func_input += "3"
            st.rerun()
        if col4.button(".", use_container_width=True, key="int_dot"):
            st.session_state.int_func_input += "."
            st.rerun()
        
        # Controls
        st.sidebar.markdown("##### Controls")
        col1, col2, col3, col4 = st.sidebar.columns(4)
        if col1.button("0", use_container_width=True, key="int_0"):
            st.session_state.int_func_input += "0"
            st.rerun()
        if col2.button("⌫", use_container_width=True, key="int_back"):
            st.session_state.int_func_input = st.session_state.int_func_input[:-1]
            st.rerun()
        if col3.button("🗑️", use_container_width=True, key="int_clear"):
            st.session_state.int_func_input = ""
            st.rerun()
        if col4.button("␣", use_container_width=True, key="int_space"):
            st.session_state.int_func_input += " "
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Validate function
        is_valid, f, error_msg = validate_function(func_str)
        
        if not is_valid:
            st.sidebar.error(f"❌ Invalid Function: {error_msg}")
            st.error(f"⚠️ **Error:** {error_msg}")
            st.info("💡 **Tip:** Use the keypad buttons or check syntax. Examples: x**2, sin(x), exp(x)")
            func_str = ""  # Prevent further processing
        
        # Integration limits and n
        st.sidebar.markdown("### 📏 Integration Parameters")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            a = st.sidebar.number_input(
                "**Lower Limit (a):**",
                value=float(st.session_state.int_a),
                format="%.6f",
                key="int_a_input",
                help="Lower bound of integration"
            )
            st.session_state.int_a = a
        with col2:
            b = st.sidebar.number_input(
                "**Upper Limit (b):**",
                value=float(st.session_state.int_b),
                format="%.6f",
                key="int_b_input",
                help="Upper bound of integration"
            )
            st.session_state.int_b = b
        
        if a >= b:
            st.sidebar.error("❌ Lower limit (a) must be less than upper limit (b)!")
        
        # Number of intervals n
        n_intervals = st.sidebar.number_input(
            "**Number of Intervals (n):**",
            min_value=1,
            max_value=1000,
            value=int(st.session_state.int_n),
            step=1,
            help="Number of subintervals. h = (b-a)/n"
        )
        st.session_state.int_n = n_intervals
        
        # Calculate h
        if a < b and n_intervals > 0:
            h_calculated = (b - a) / n_intervals
            st.sidebar.success(f"✅ **Step size:** h = (b-a)/n = ({b:.4f} - {a:.4f})/{n_intervals} = {h_calculated:.6f}")
            
            # Calculate number of points
            num_points_from_n = n_intervals + 1
            
            # Validate point count for selected method
            if method_key == 'simpson_1_3':
                if num_points_from_n % 2 == 0:
                    st.sidebar.warning(f"⚠️ Simpson's 1/3 needs ODD points. With n={n_intervals}, you'll have {num_points_from_n} points. Consider n={n_intervals+1} or n={n_intervals-1}.")
            elif method_key == 'simpson_3_8':
                if n_intervals % 3 != 0:
                    st.sidebar.warning(f"⚠️ Simpson's 3/8 needs n divisible by 3. You have n={n_intervals}. Consider n={3*(n_intervals//3)} or n={3*((n_intervals//3)+1)}.")
            
            # Generate points if function is valid
            if is_valid and func_str:
                x_points = np.linspace(a, b, num_points_from_n).tolist()
                try:
                    y_points = [float(f(x)) for x in x_points]
                    st.sidebar.success(f"✅ Generated {num_points_from_n} points from function")
                except Exception as e:
                    st.sidebar.error(f"❌ Error evaluating function: {str(e)}")
                    x_points = []
                    y_points = []
            else:
                x_points = []
                y_points = []
        else:
            x_points = []
            y_points = []
            h_calculated = 0.0
            n_intervals = 0
            is_valid = False
            func_str = ""
            n_intervals = 0
    
    else:  # Manual Entry
        st.sidebar.markdown("### 📍 Enter Data Points")
        st.sidebar.info("💡 **Note:** x values must be equally spaced. The spacing h will be calculated automatically.")
        
        # Default example data (x² from 0 to 4)
        default_example = {
            'x': [0.0, 1.0, 2.0, 3.0, 4.0],
            'y': [0.0, 1.0, 4.0, 9.0, 16.0]
        }
        
        # Load default button
        if st.sidebar.button("📥 Load Default Example (x²)", use_container_width=True):
            for i in range(min(num_points, len(default_example['x']))):
                st.session_state.int_x_points[i] = default_example['x'][i] if i < len(default_example['x']) else i * 1.0
                st.session_state.int_y_points[i] = default_example['y'][i] if i < len(default_example['y']) else (i * 1.0) ** 2
            st.sidebar.success("✅ Default values loaded!")
            st.rerun()
        
        # Input points
        x_points = []
        y_points = []
        
        for i in range(num_points):
            col1, col2 = st.sidebar.columns(2)
            with col1:
                default_x = float(st.session_state.int_x_points[i]) if i < len(st.session_state.int_x_points) else (float(default_example['x'][i]) if i < len(default_example['x']) else i * 1.0)
                x_val = col1.number_input(
                    f"x{i}",
                    value=default_x,
                    format="%.6f",
                    key=f"int_x_{i}",
                    help=f"x-coordinate for point {i}"
                )
            with col2:
                default_y = float(st.session_state.int_y_points[i]) if i < len(st.session_state.int_y_points) else (float(default_example['y'][i]) if i < len(default_example['y']) else 0.0)
                y_val = col2.number_input(
                    f"y{i}",
                    value=default_y,
                    format="%.6f",
                    key=f"int_y_{i}",
                    help=f"y-coordinate (function value) for point {i}"
                )
            x_points.append(x_val)
            y_points.append(y_val)
            st.session_state.int_x_points[i] = x_val
            st.session_state.int_y_points[i] = y_val
    
    st.sidebar.markdown("---")
    
    # Calculate button
    calculate_integration = st.sidebar.button(
        "🚀 INTEGRATE",
        type="primary",
        use_container_width=True
    )
    
    # Main content
    st.markdown("## 📐 Numerical Integration")
    st.markdown(f"### Using **{integration_method}**")
    
    if calculate_integration:
        try:
            # For function input, ensure points were generated
            if input_method == "Function Input":
                if 'is_valid' in locals() and not is_valid:
                    st.error("❌ Please enter a valid function first!")
                    st.stop()
                if len(x_points) == 0 or len(y_points) == 0:
                    st.error("❌ Could not generate points from function. Check your function and limits.")
                    st.stop()
            
            # Validate data
            if len(x_points) != len(y_points):
                st.error("❌ Number of x and y values must match!")
                st.stop()
            
            if len(x_points) < 2:
                st.error("❌ Need at least 2 data points!")
                st.stop()
            
            # Perform integration
            with st.spinner("Calculating integral..."):
                result = numerical_integration(
                    x_points=x_points,
                    y_points=y_points,
                    method=method_key
                )
            
            if not result['success']:
                st.error(f"❌ {result['message']}")
                st.stop()
            
            # Display result
            st.success(result['message'])
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Integral Value", format_max_precision(result['integral']))
            with col2:
                st.metric("Interval", f"[{format_max_precision(result['interval'][0])}, {format_max_precision(result['interval'][1])}]")
            with col3:
                st.metric("Points Used", result['num_points'])
            with col4:
                st.metric("Spacing (h)", format_max_precision(result['spacing_h']))
            
            # Show function info if function input was used
            if input_method == "Function Input" and 'is_valid' in locals() and is_valid and 'func_str' in locals() and func_str:
                st.info(f"📐 **Function:** f(x) = {func_str} | **Intervals:** n = {n_intervals if 'n_intervals' in locals() else 'N/A'} | **h = (b-a)/n = {format_max_precision(result['spacing_h'])}**")
            
            st.markdown("---")
            
            # Formula display
            st.markdown("### 📐 Formula Used")
            if result.get('formula_latex'):
                st.latex(result['formula_latex'])
            else:
                st.code(result['formula_used'], language="text")
            
            st.markdown("---")
            
            # Detailed breakdown
            st.markdown("### 🧮 Step-by-Step Calculation")
            
            breakdown = result['breakdown']
            h = breakdown['h']
            method_name = result['method_name']
            
            with st.expander("📊 **View Complete Breakdown**", expanded=True):
                st.markdown(f"**Spacing:** h = {format_max_precision(h)}")
                st.markdown(f"**Interval:** [{format_max_precision(breakdown['interval'][0])}, {format_max_precision(breakdown['interval'][1])}]")
                st.markdown(f"**Number of intervals:** n = {result['num_intervals']}")
                st.markdown(f"**Number of points:** {result['num_points']}")
                
                # Create table
                breakdown_data = []
                for term in breakdown['terms']:
                    breakdown_data.append({
                        'i': term['index'],
                        'xᵢ': format_max_precision(term['x']),
                        'yᵢ': format_max_precision(term['y']),
                        'Coefficient': format_max_precision(term['coefficient']),
                        'Contribution': format_max_precision(term['contribution'])
                    })
                
                df = pd.DataFrame(breakdown_data)
                
                # Highlight cells based on coefficient
                def highlight_coefficient(row):
                    styles = [''] * len(row)
                    try:
                        coeff = float(row['Coefficient'])
                        if abs(coeff - 4.0) < 0.01 or abs(coeff - 3.0) < 0.01:
                            return ['', '', '', 'background-color: #ffd700; font-weight: bold;', '']
                        elif abs(coeff - 2.0) < 0.01:
                            return ['', '', '', 'background-color: #90ee90; font-weight: bold;', '']
                        elif abs(coeff - 0.5) < 0.01:
                            return ['', '', '', 'background-color: #87ceeb; font-weight: bold;', '']
                    except:
                        pass
                    return styles
                
                styled_df = df.style.apply(highlight_coefficient, axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                st.caption("💡 **Color coding:** Gold = 4 or 3, Green = 2, Blue = 0.5")
                
                # Show sum
                total = sum([term['contribution'] for term in breakdown['terms']])
                st.success(f"**Total Integral = Σ(Contributions) = {format_max_precision(total)}**")
            
            # Detailed Step-by-Step Simplification
            st.markdown("---")
            with st.expander("🔢 **Detailed Step-by-Step Simplification**", expanded=True):
                st.markdown("#### Step 1: Identify Parameters")
                st.markdown(f"- **Step size:** h = {format_max_precision(h)}")
                st.markdown(f"- **Interval:** [{format_max_precision(breakdown['interval'][0])}, {format_max_precision(breakdown['interval'][1])}]")
                st.markdown(f"- **Number of points:** {result['num_points']}")
                st.markdown(f"- **Number of intervals:** n = {result['num_intervals']}")
                
                st.markdown("---")
                st.markdown("#### Step 2: Apply the Formula")
                
                if method_key == 'trapezoidal':
                    st.markdown(f"**Trapezoidal Rule Formula:**")
                    st.latex(r"I = \frac{h}{2}[y_0 + 2y_1 + 2y_2 + \cdots + 2y_{n-1} + y_n]")
                    
                    st.markdown("**Step 2.1: Identify Terms**")
                    first_term = breakdown['terms'][0]
                    last_term = breakdown['terms'][-1]
                    middle_terms = breakdown['terms'][1:-1]
                    
                    st.markdown(f"- **First term:** y₀ = {format_max_precision(first_term['y'])} (coefficient = 0.5)")
                    st.markdown(f"- **Middle terms:** {len(middle_terms)} terms with coefficient = 1.0")
                    st.markdown(f"- **Last term:** yₙ = {format_max_precision(last_term['y'])} (coefficient = 0.5)")
                    
                    st.markdown("**Step 2.2: Calculate Each Term**")
                    st.markdown(f"**First term contribution:**")
                    st.latex(f"\\frac{{h}}{{2}} \\times y_0 = \\frac{{{format_max_precision(h)}}}{{2}} \\times {format_max_precision(first_term['y'])} = {format_max_precision(first_term['contribution'])}")
                    
                    st.markdown("**Middle terms contributions:**")
                    middle_sum = sum([t['y'] for t in middle_terms])
                    middle_contrib_sum = sum([t['contribution'] for t in middle_terms])
                    st.latex(f"h \\times (y_1 + y_2 + \\cdots + y_{{{len(middle_terms)}}}) = {format_max_precision(h)} \\times {format_max_precision(middle_sum)} = {format_max_precision(middle_contrib_sum)}")
                    
                    st.markdown("**Last term contribution:**")
                    st.latex(f"\\frac{{h}}{{2}} \\times y_n = \\frac{{{format_max_precision(h)}}}{{2}} \\times {format_max_precision(last_term['y'])} = {format_max_precision(last_term['contribution'])}")
                    
                    st.markdown("**Step 2.3: Sum All Contributions**")
                    st.latex(f"I = {format_max_precision(first_term['contribution'])} + {format_max_precision(middle_contrib_sum)} + {format_max_precision(last_term['contribution'])} = {format_max_precision(result['integral'])}")
                
                elif method_key == 'simpson_1_3':
                    st.markdown(f"**Simpson's 1/3 Rule Formula:**")
                    st.latex(r"I = \frac{h}{3}[y_0 + 4y_1 + 2y_2 + 4y_3 + 2y_4 + \cdots + 4y_{n-1} + y_n]")
                    
                    st.markdown("**Step 2.1: Group Terms by Coefficient**")
                    first_term = breakdown['terms'][0]
                    last_term = breakdown['terms'][-1]
                    odd_terms = [t for t in breakdown['terms'][1:-1] if t['index'] % 2 == 1]
                    even_terms = [t for t in breakdown['terms'][1:-1] if t['index'] % 2 == 0]
                    
                    st.markdown(f"- **First term:** y₀ = {format_max_precision(first_term['y'])} (coefficient = 1)")
                    st.markdown(f"- **Odd-indexed terms:** {len(odd_terms)} terms with coefficient = 4")
                    st.markdown(f"- **Even-indexed terms:** {len(even_terms)} terms with coefficient = 2")
                    st.markdown(f"- **Last term:** yₙ = {format_max_precision(last_term['y'])} (coefficient = 1)")
                    
                    st.markdown("**Step 2.2: Calculate Each Group**")
                    st.markdown("**First term:**")
                    st.latex(f"\\frac{{h}}{{3}} \\times y_0 = \\frac{{{format_max_precision(h)}}}{{3}} \\times {format_max_precision(first_term['y'])} = {format_max_precision(first_term['contribution'])}")
                    
                    if odd_terms:
                        odd_sum_y = sum([t['y'] for t in odd_terms])
                        odd_contrib_sum = sum([t['contribution'] for t in odd_terms])
                        odd_indices = ", ".join([f"y_{{{t['index']}}}" for t in odd_terms[:5]])
                        if len(odd_terms) > 5:
                            odd_indices += f", \\ldots, y_{{{odd_terms[-1]['index']}}}"
                        st.markdown("**Odd-indexed terms (coefficient 4):**")
                        st.latex(f"\\frac{{h}}{{3}} \\times 4 \\times ({odd_indices}) = \\frac{{{format_max_precision(h)}}}{{3}} \\times 4 \\times {format_max_precision(odd_sum_y)} = {format_max_precision(odd_contrib_sum)}")
                    
                    if even_terms:
                        even_sum_y = sum([t['y'] for t in even_terms])
                        even_contrib_sum = sum([t['contribution'] for t in even_terms])
                        even_indices = ", ".join([f"y_{{{t['index']}}}" for t in even_terms[:5]])
                        if len(even_terms) > 5:
                            even_indices += f", \\ldots, y_{{{even_terms[-1]['index']}}}"
                        st.markdown("**Even-indexed terms (coefficient 2):**")
                        st.latex(f"\\frac{{h}}{{3}} \\times 2 \\times ({even_indices}) = \\frac{{{format_max_precision(h)}}}{{3}} \\times 2 \\times {format_max_precision(even_sum_y)} = {format_max_precision(even_contrib_sum)}")
                    
                    st.markdown("**Last term:**")
                    st.latex(f"\\frac{{h}}{{3}} \\times y_n = \\frac{{{format_max_precision(h)}}}{{3}} \\times {format_max_precision(last_term['y'])} = {format_max_precision(last_term['contribution'])}")
                    
                    st.markdown("**Step 2.3: Sum All Contributions**")
                    parts = [format_max_precision(first_term['contribution'])]
                    if odd_contrib_sum > 0:
                        parts.append(format_max_precision(odd_contrib_sum))
                    if even_contrib_sum > 0:
                        parts.append(format_max_precision(even_contrib_sum))
                    parts.append(format_max_precision(last_term['contribution']))
                    st.latex(f"I = {' + '.join(parts)} = {format_max_precision(result['integral'])}")
                
                elif method_key == 'simpson_3_8':
                    st.markdown(f"**Simpson's 3/8 Rule Formula:**")
                    st.latex(r"I = \frac{3h}{8}[y_0 + 3y_1 + 3y_2 + 2y_3 + 3y_4 + 3y_5 + 2y_6 + \cdots + 3y_{n-1} + y_n]")
                    
                    st.markdown("**Step 2.1: Group Terms by Coefficient**")
                    first_term = breakdown['terms'][0]
                    last_term = breakdown['terms'][-1]
                    coeff_3_terms = [t for t in breakdown['terms'][1:-1] if t['coefficient'] == 3]
                    coeff_2_terms = [t for t in breakdown['terms'][1:-1] if t['coefficient'] == 2]
                    
                    st.markdown(f"- **First term:** y₀ = {format_max_precision(first_term['y'])} (coefficient = 1)")
                    st.markdown(f"- **Terms with coefficient 3:** {len(coeff_3_terms)} terms")
                    st.markdown(f"- **Terms with coefficient 2:** {len(coeff_2_terms)} terms (boundaries)")
                    st.markdown(f"- **Last term:** yₙ = {format_max_precision(last_term['y'])} (coefficient = 1)")
                    
                    st.markdown("**Step 2.2: Calculate Each Group**")
                    st.markdown("**First term:**")
                    st.latex(f"\\frac{{3h}}{{8}} \\times y_0 = \\frac{{3 \\times {format_max_precision(h)}}}{{8}} \\times {format_max_precision(first_term['y'])} = {format_max_precision(first_term['contribution'])}")
                    
                    if coeff_3_terms:
                        coeff_3_sum_y = sum([t['y'] for t in coeff_3_terms])
                        coeff_3_contrib_sum = sum([t['contribution'] for t in coeff_3_terms])
                        coeff_3_indices = ", ".join([f"y_{{{t['index']}}}" for t in coeff_3_terms[:5]])
                        if len(coeff_3_terms) > 5:
                            coeff_3_indices += f", \\ldots, y_{{{coeff_3_terms[-1]['index']}}}"
                        st.markdown("**Terms with coefficient 3:**")
                        st.latex(f"\\frac{{3h}}{{8}} \\times 3 \\times ({coeff_3_indices}) = \\frac{{3 \\times {format_max_precision(h)}}}{{8}} \\times 3 \\times {format_max_precision(coeff_3_sum_y)} = {format_max_precision(coeff_3_contrib_sum)}")
                    
                    if coeff_2_terms:
                        coeff_2_sum_y = sum([t['y'] for t in coeff_2_terms])
                        coeff_2_contrib_sum = sum([t['contribution'] for t in coeff_2_terms])
                        coeff_2_indices = ", ".join([f"y_{{{t['index']}}}" for t in coeff_2_terms[:5]])
                        if len(coeff_2_terms) > 5:
                            coeff_2_indices += f", \\ldots, y_{{{coeff_2_terms[-1]['index']}}}"
                        st.markdown("**Terms with coefficient 2 (boundaries):**")
                        st.latex(f"\\frac{{3h}}{{8}} \\times 2 \\times ({coeff_2_indices}) = \\frac{{3 \\times {format_max_precision(h)}}}{{8}} \\times 2 \\times {format_max_precision(coeff_2_sum_y)} = {format_max_precision(coeff_2_contrib_sum)}")
                    
                    st.markdown("**Last term:**")
                    st.latex(f"\\frac{{3h}}{{8}} \\times y_n = \\frac{{3 \\times {format_max_precision(h)}}}{{8}} \\times {format_max_precision(last_term['y'])} = {format_max_precision(last_term['contribution'])}")
                    
                    st.markdown("**Step 2.3: Sum All Contributions**")
                    parts = [format_max_precision(first_term['contribution'])]
                    if coeff_3_terms:
                        parts.append(format_max_precision(coeff_3_contrib_sum))
                    if coeff_2_terms:
                        parts.append(format_max_precision(coeff_2_contrib_sum))
                    parts.append(format_max_precision(last_term['contribution']))
                    st.latex(f"I = {' + '.join(parts)} = {format_max_precision(result['integral'])}")
                
                st.markdown("---")
                st.markdown("#### Step 3: Final Answer")
                st.success(f"**∫f(x)dx ≈ {format_max_precision(result['integral'])}**")
            
            # Visualization
            st.markdown("---")
            st.markdown("### 📈 Visualization")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot points
            ax.plot(x_points, y_points, 'ro-', linewidth=2, markersize=8, label='Data Points')
            
            # Fill area under curve
            ax.fill_between(x_points, 0, y_points, alpha=0.3, color='skyblue', label=f'Integral ≈ {format_max_precision(result["integral"])}')
            
            # Show method-specific visualization
            if method_key == 'trapezoidal':
                # Draw trapezoids
                for i in range(len(x_points) - 1):
                    xs = [x_points[i], x_points[i+1], x_points[i+1], x_points[i], x_points[i]]
                    ys = [0, 0, y_points[i+1], y_points[i], 0]
                    ax.plot(xs, ys, 'b--', alpha=0.5, linewidth=1)
            
            ax.axhline(0, color='black', linewidth=0.5)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x', fontsize=12, fontweight='bold')
            ax.set_ylabel('y = f(x)', fontsize=12, fontweight='bold')
            ax.set_title(f'Numerical Integration: {integration_method}', fontsize=14, fontweight='bold')
            ax.legend()
            st.pyplot(fig)
            
            # Data table
            st.markdown("---")
            st.markdown("### 📊 Input Data")
            
            data_df = pd.DataFrame({
                'i': range(len(x_points)),
                'x': [format_max_precision(x) for x in x_points],
                'y': [format_max_precision(y) for y in y_points]
            })
            st.dataframe(data_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"❌ Error calculating integral: {str(e)}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
    
    else:
        # Welcome screen
        st.info("👈 **Get Started:** Enter data points in the sidebar and click 'INTEGRATE'")
        
        st.markdown("### 📚 About Numerical Integration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Trapezoidal Rule:**
            - Approximates area using trapezoids
            - Works with any number of points (≥2)
            - Simple but less accurate
            - Error: O(h²)
            """)
            
            st.latex(r"\int_a^b f(x)dx \approx \frac{h}{2}[y_0 + 2y_1 + 2y_2 + \cdots + 2y_{n-1} + y_n]")
        
        with col2:
            st.markdown("""
            **Simpson's Rules:**
            - More accurate polynomial approximations
            - 1/3 Rule: Uses parabolas (requires ODD points)
            - 3/8 Rule: Uses cubics (requires n divisible by 3)
            - Error: O(h⁴)
            """)
            
            st.latex(r"\int_a^b f(x)dx \approx \frac{h}{3}[y_0 + 4y_1 + 2y_2 + 4y_3 + \cdots + y_n]")
        
        st.markdown("---")
        st.markdown("### 🔬 Method Comparison")
        
        comparison_table = pd.DataFrame({
            'Method': ['Trapezoidal Rule', "Simpson's 1/3 Rule", "Simpson's 3/8 Rule"],
            'Minimum Points': ['2', '3 (ODD)', '4'],
            'Accuracy': ['O(h²)', 'O(h⁴)', 'O(h⁴)'],
            'Best For': ['Any data', 'Smooth functions', 'Very smooth functions'],
            'Complexity': ['Simple', 'Medium', 'Medium']
        })
        
        st.dataframe(comparison_table, use_container_width=True, hide_index=True)

st.caption("Numerical Computing Project | Root Finding & Interpolation Calculator")
