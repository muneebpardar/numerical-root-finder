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
from utils.validators import validate_function, validate_interval, preprocess_function

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
    /* Force header visibility in both light and dark mode */
    .force-header h1 {
        color: white !important;
        -webkit-text-fill-color: white !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
        position: relative;
        z-index: 9999;
    }
    
    /* Function input styling - works in both modes */
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        font-size: 18px !important;
        font-weight: 600;
        letter-spacing: 0.5px;
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Light mode function input */
    [data-theme="light"] .stTextArea textarea {
        color: #1f4788;
        background-color: #f8f9fa;
    }
    
    /* Dark mode function input */
    [data-theme="dark"] .stTextArea textarea {
        color: #e0e0e0;
        background-color: #2b2b2b;
    }
    
    /* Button styling */
    .stButton button {
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Primary button (Calculate) */
    .stButton button[kind="primary"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-size: 16px;
        font-weight: 700;
        border: none;
    }
    
    /* Title styling - responsive to theme */
    h1 {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Dark mode title fix */
    [data-theme="dark"] h1 {
        background: linear-gradient(90deg, #8b9eea 0%, #9b6bc2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Metric cards - theme aware */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 600;
    }
    
    /* Table styling - works in both themes */
    .dataframe {
        font-size: 13px;
    }
    
    .dataframe thead tr th {
        background-color: #667eea !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    
    /* Sidebar styling - theme aware */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    [data-theme="dark"] [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e1e 0%, #2b2b2b 100%);
    }
    
    /* Success boxes - visible in dark mode */
    .stSuccess {
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
    }
    
    [data-theme="light"] .stSuccess {
        background-color: #d4edda;
        color: #155724;
    }
    
    [data-theme="dark"] .stSuccess {
        background-color: #1e4d2b;
        color: #a8e6a3;
    }
    
    /* Error boxes - visible in dark mode */
    .stError {
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
    }
    
    [data-theme="light"] .stError {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    [data-theme="dark"] .stError {
        background-color: #4d1f1f;
        color: #f8a1a1;
    }
    
    /* Info boxes - visible in dark mode */
    .stInfo {
        border-left: 5px solid #0c5460;
        padding: 15px;
        border-radius: 5px;
    }
    
    [data-theme="light"] .stInfo {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    
    [data-theme="dark"] .stInfo {
        background-color: #1a3d42;
        color: #a8d8e0;
    }
    
    /* Warning boxes - visible in dark mode */
    .stWarning {
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
    }
    
    [data-theme="light"] .stWarning {
        background-color: #fff3cd;
        color: #856404;
    }
    
    [data-theme="dark"] .stWarning {
        background-color: #4d4020;
        color: #ffe599;
    }
    
    /* Expander styling - theme aware */
    .streamlit-expanderHeader {
        border-radius: 8px;
        font-weight: 600;
    }
    
    [data-theme="light"] .streamlit-expanderHeader {
        background-color: #f0f2f6;
    }
    
    [data-theme="dark"] .streamlit-expanderHeader {
        background-color: #2b2b2b;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Enhanced Radio buttons - theme aware */
    .stRadio > div {
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    [data-theme="light"] .stRadio > div {
        background: white;
    }
    
    [data-theme="dark"] .stRadio > div {
        background: #2b2b2b;
    }
    
    /* Download button enhancement */
    .stDownloadButton button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
        color: white !important;
    }
    
    /* Code blocks - dark mode fix */
    [data-theme="dark"] code {
        color: #e0e0e0 !important;
        background-color: #2b2b2b !important;
    }
    
    /* LaTeX in dark mode */
    [data-theme="dark"] .katex {
        color: #e0e0e0 !important;
    }
    
    /* Dataframe in dark mode */
    [data-theme="dark"] .dataframe {
        color: #e0e0e0;
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
st.sidebar.header("⚙️ Configuration")

problem_type = st.sidebar.radio(
    "**📚 Select Problem Type:**",
    ["🎯 Root Finding", "📊 Lagrange Interpolation", "🔢 Divided Difference Interpolation", 
     "🔧 Linear Systems (Jacobi)", "⚡ Linear Systems (Gauss-Seidel)"],
    key="problem_type"
)

st.sidebar.markdown("---")

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

    if 'func_input' not in st.session_state:
        st.session_state.func_input = "x**3 - x - 2"

    func_str = st.sidebar.text_area(
        "Enter f(x):",
        value=st.session_state.func_input,
        height=100,
        help="Build your function using the keypad below",
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

    # ============================================================================
    # ADD THIS SECTION TO YOUR APP.PY AFTER THE REGULAR TRIG FUNCTIONS
    # Replace the "Advanced" section with this enhanced version
    # ============================================================================

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
        
        # ADD THIS NEW SECTION - Shows preprocessed function
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
        x0 = st.sidebar.number_input("Initial guess (x₀):", value=1.5, format="%.10f")

    # Secant Method
    elif method == "Secant Method":
        col1, col2 = st.sidebar.columns(2)
        x0 = col1.number_input("First guess (x₀):", value=1.0, format="%.10f")
        x1 = col2.number_input("Second guess (x₁):", value=2.0, format="%.10f")

    # Fixed Point
    elif method == "Fixed Point Method":
        # Interval input for convergence check / plotting
        col1, col2 = st.sidebar.columns(2)
        a_str = col1.text_input("a (left, for check/plot):", value="0")
        b_str = col2.text_input("b (right, for check/plot):", value="2*pi")
        
        try:
            a = float(sympify(a_str))
            b = float(sympify(b_str))
        except:
            st.sidebar.error("❌ Invalid interval. Use expressions like 0 or 2*pi.")
            st.stop()
        
        # Initial guess
        x0 = st.sidebar.number_input("Initial guess (x₀):", value=1.5, format="%.10f")
        
        # g(x) input
        st.sidebar.info("💡 Transform f(x)=0 to x=g(x). Example: x³-x-2 → g(x)=(x+2)^(1/3)")
        g_str = st.sidebar.text_input(
            "g(x) function:", 
            value="(x + 2)**(1/3)", 
            help="Example: For x³-x-2=0 → x³=x+2 → x=(x+2)^(1/3)"
        )
        
        # Validate g(x)
        is_valid_g, g, error_msg_g = validate_function(g_str)
        if not is_valid_g:
            st.sidebar.error(f"❌ Invalid g(x)")
            st.sidebar.warning(error_msg_g)
        else:
            st.sidebar.success("✅ Valid g(x)")

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
                
                with st.spinner("Calculating..."):
                    result = newton_raphson(f, f_prime, x0, tol=tolerance, max_iter=max_iter)
            
            elif method == "Secant Method":
                with st.spinner("Calculating..."):
                    result = secant(f, x0, x1, tol=tolerance, max_iter=max_iter)
            
            elif method == "Fixed Point Method":
                if not is_valid_g:
                    st.error(f"❌ Invalid g(x): {error_msg_g}")
                    st.stop()
                
                with st.spinner("Calculating..."):
                    result = fixed_point(g, x0, tol=tolerance, max_iter=max_iter)
            
            # Display results
            if result['success']:
                st.success(f"✅ {result['message']}")
                
                # ============================================================================
                # COMPLETE DIFFERENTIATION INTEGRATION FOR APP.PY
                # Insert these code blocks in the appropriate sections of your app.py
                # ============================================================================

                # ============================================================================
                # PART 1: NEWTON-RAPHSON DETAILED DIFFERENTIATION
                # Insert this AFTER the success message and BEFORE the metrics display
                # (Around line 850-900 in your current app.py)
                # ============================================================================

                # After: if result['success']:
                #        st.success(f"✅ {result['message']}")
                # ADD THIS:

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


                # ============================================================================
                # PART 2: SECANT METHOD DERIVATIVE APPROXIMATION
                # Insert this AFTER the Secant method result display
                # ============================================================================

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
                # PART 3: FIXED POINT METHOD DERIVATIVE ANALYSIS
                # Insert this AFTER Fixed Point method result display
                # ============================================================================

                elif method == "Fixed Point Method":
                    st.markdown("---")
                    st.markdown("### 📊 Convergence Analysis: Role of g'(x)")
                    
                    with st.expander("📐 **Why g'(x) Determines Convergence**", expanded=True):
                        st.markdown("""
                        The Fixed Point method transforms **f(x) = 0** into **x = g(x)** and iterates:
                        """)
                        st.latex(r"x_{n+1} = g(x_n)")
                        
                        st.markdown("""
                        #### Convergence Theorem:
                        
                        The method converges to root α if and only if:
                        """)
                        st.latex(r"|g'(\alpha)| < 1")
                        
                        st.markdown("**Why?** Let's analyze the error at each step.")
                        
                        st.markdown("---")
                        st.markdown("#### Error Analysis:")
                        
                        st.markdown("Let eₙ = xₙ - α (error at iteration n)")
                        st.markdown("**Step 1:** Start with iteration formula")
                        st.latex(r"x_{n+1} = g(x_n)")
                        
                        st.markdown("**Step 2:** Since α is the true root: α = g(α)")
                        st.markdown("**Step 3:** Subtract these equations:")
                        st.latex(r"x_{n+1} - \alpha = g(x_n) - g(\alpha)")
                        st.latex(r"e_{n+1} = g(x_n) - g(\alpha)")
                        
                        st.markdown("**Step 4:** Apply Mean Value Theorem:")
                        st.latex(r"g(x_n) - g(\alpha) = g'(\xi)(x_n - \alpha) \quad \text{for some } \xi \in [x_n, \alpha]")
                        
                        st.markdown("**Step 5:** Therefore:")
                        st.latex(r"e_{n+1} = g'(\xi) \cdot e_n")
                        
                        st.markdown("**Conclusion:**")
                        st.latex(r"|e_{n+1}| \approx |g'(\alpha)| \cdot |e_n|")
                        
                        st.markdown("""
                        - If **|g'(α)| < 1**: Error decreases → **Converges** ✅
                        - If **|g'(α)| > 1**: Error increases → **Diverges** ❌
                        - If **|g'(α)| = 1**: **Boundary case** (may or may not converge)
                        """)
                    
                    with st.expander("🔬 **Calculate g'(x) for Your Function**", expanded=True):
                        st.markdown("#### Your iteration function:")
                        st.code(f"g(x) = {g_str}", language="python")
                        
                        try:
                            from sympy import symbols, sympify, diff, latex, simplify, Abs
                            from utils.validators import preprocess_function
                            
                            x_sym = symbols('x')
                            processed_g = preprocess_function(g_str)
                            g_expr = sympify(processed_g)
                            
                            st.markdown("**Mathematical form:**")
                            st.latex(f"g(x) = {latex(g_expr)}")
                            
                            st.markdown("---")
                            st.markdown("#### Computing g'(x):")
                            
                            g_prime_expr = diff(g_expr, x_sym)
                            
                            st.latex(f"\\frac{{d}}{{dx}}\\left[{latex(g_expr)}\\right] = {latex(g_prime_expr)}")
                            
                            simplified_g_prime = simplify(g_prime_expr)
                            
                            if simplified_g_prime != g_prime_expr:
                                st.markdown("**Simplified:**")
                                st.latex(f"g'(x) = {latex(simplified_g_prime)}")
                            else:
                                st.markdown("**Derivative:**")
                                st.latex(f"g'(x) = {latex(g_prime_expr)}")
                            
                            st.markdown("---")
                            
                            # Evaluate g' at iterations
                            st.markdown("#### Values of g'(x) During Iteration:")
                            
                            g_prime_func = lambdify(x_sym, simplified_g_prime, 'numpy')
                            
                            gprime_table = []
                            for iteration in result['iterations'][:min(10, len(result['iterations']))]:
                                x_val = iteration['xₙ']
                                
                                try:
                                    gprime_val = float(g_prime_func(x_val))
                                    abs_gprime = abs(gprime_val)
                                    
                                    if abs_gprime < 1:
                                        status = "✅ Converging"
                                    elif abs_gprime > 1:
                                        status = "❌ Diverging"
                                    else:
                                        status = "⚠️ Boundary"
                                    
                                    gprime_table.append({
                                        'n': iteration['n'],
                                        'xₙ': f"{x_val:.6f}",
                                        "g'(xₙ)": f"{gprime_val:.6f}",
                                        "|g'(xₙ)|": f"{abs_gprime:.6f}",
                                        'Status': status
                                    })
                                except:
                                    pass
                            
                            st.dataframe(pd.DataFrame(gprime_table), use_container_width=True, hide_index=True)
                            
                            # Final analysis
                            if result['success']:
                                final_x = result['root']
                                final_gprime = float(g_prime_func(final_x))
                                
                                st.markdown("---")
                                st.markdown("#### Convergence Assessment at Root:")
                                
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Root α", f"{final_x:.6f}")
                                col2.metric("g'(α)", f"{final_gprime:.6f}")
                                col3.metric("|g'(α)|", f"{abs(final_gprime):.6f}")
                                
                                if abs(final_gprime) < 0.5:
                                    st.success(f"✅ Excellent! |g'(α)| = {abs(final_gprime):.4f} < 0.5 → Fast convergence")
                                elif abs(final_gprime) < 1:
                                    st.success(f"✅ Good! |g'(α)| = {abs(final_gprime):.4f} < 1 → Method converges")
                                elif abs(final_gprime) == 1:
                                    st.warning(f"⚠️ Boundary case: |g'(α)| = 1 → Convergence uncertain")
                                else:
                                    st.error(f"❌ Problem: |g'(α)| = {abs(final_gprime):.4f} > 1 → Should not converge (may be luck or wrong root)")
                            
                        except Exception as e:
                            st.error(f"Could not compute g'(x): {e}")
                            st.info("Try simplifying your g(x) function.")
                    
                    with st.expander("💡 **How to Choose Good g(x)**", expanded=False):
                        st.markdown("""
                        #### Guidelines for Choosing g(x):
                        
                        Given **f(x) = 0**, there are multiple ways to write **x = g(x)**:
                        
                        **Example:** f(x) = x³ - x - 2 = 0
                        
                        **Option 1:** x = x³ - 2
                        """)
                        st.latex(r"g_1(x) = x^3 - 2, \quad g_1'(x) = 3x^2")
                        st.markdown("At root x ≈ 1.52: |g₁'(1.52)| ≈ 6.9 > 1 ❌ **Diverges!**")
                        
                        st.markdown("**Option 2:** x = (x + 2)^(1/3)")
                        st.latex(r"g_2(x) = (x+2)^{1/3}, \quad g_2'(x) = \frac{1}{3(x+2)^{2/3}}")
                        st.markdown("At root x ≈ 1.52: |g₂'(1.52)| ≈ 0.16 < 1 ✅ **Converges!**")
                        
                        st.markdown("---")
                        st.markdown("#### Strategy:")
                        st.markdown("""
                        1. **Rearrange** f(x) = 0 into x = g(x) in multiple ways
                        2. **Compute** g'(x) for each rearrangement
                        3. **Choose** the g(x) where |g'(α)| is smallest near the root
                        4. **Test** with initial guess to verify convergence
                        
                        **Common Techniques:**
                        """)
                        
                        st.code("""
                # If f(x) = x² - a = 0, solve for x = a/x
                # If f(x) = x - cos(x) = 0, use x = cos(x)
                # If f(x) = x³ - x - 2 = 0, use x = (x+2)^(1/3)
                        """, language="python")
                        
                        st.markdown("""
                        **Pro Tip:** For f(x) = 0, try:
                        """)
                        st.latex(r"x = x - \lambda f(x) \quad \text{where } \lambda \text{ is chosen so } |g'(x)| < 1")


                # ============================================================================
                # PART 4: GENERAL DERIVATIVE DISPLAY HELPER (for all methods)
                # Add this section after importing libraries at the top
                # ============================================================================

                # Insert this near the top of app.py after imports
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


                # ============================================================================
                # PART 5: COMPARISON MODE - Add derivative comparison
                # Insert this in the "Compare All Methods" section
                # ============================================================================

                # In the Compare All Methods section, after showing the comparison table,
                # add this section:

                if method == "🔬 Compare All Methods" and calculate:
                    # ... existing comparison code ...
                    
                    # ADD THIS NEW SECTION
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


                # ============================================================================
                # PART 6: Enhanced iteration display with derivative insights
                # Add this to enhance the iteration table display for Newton-Raphson
                # ============================================================================

                # In Newton-Raphson section, after the iteration table, add:

                if method == "Newton-Raphson Method" and calculate and result['success']:
                    # ... existing iteration table code ...
                    
                    # ADD THIS ANALYSIS
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
                        st.markdown("#### Step Sizes (|f/f'|)")
                        
                        # Calculate step sizes
                        step_sizes = []
                        for it in result['iterations']:
                            if it["f'(xₙ)"] != 0:
                                step = abs(it['f(xₙ)'] / it["f'(xₙ)"])
                                step_sizes.append(step)
                            else:
                                step_sizes.append(0)
                        
                        fig2, ax2 = plt.subplots(figsize=(8, 5))
                        ax2.plot(iterations_list, step_sizes, 'r-s', linewidth=2, markersize=8)
                        ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
                        ax2.set_ylabel('Step Size', fontsize=12, fontweight='bold')
                        ax2.set_title('Newton Step Sizes', fontsize=14, fontweight='bold')
                        ax2.grid(True, alpha=0.3)
                        ax2.set_yscale('log')
                        
                        st.pyplot(fig2)
                        
                        st.caption("📊 Shows how far Newton-Raphson moves at each iteration")
                    
                    # Insights
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


                # ============================================================================
                # PART 7: Add a "Derivative Calculator" utility in sidebar
                # Insert this in the sidebar, perhaps as an optional tool
                # ============================================================================

                # Add this to sidebar after method selection (optional feature)
                if problem_type == "🎯 Root Finding":
                    st.sidebar.markdown("---")
                    with st.sidebar.expander("🔧 **Derivative Calculator Tool**", expanded=False):
                        st.markdown("Quick derivative calculator:")
                        
                        calc_func = st.text_input("Function:", value="x**2", key="deriv_calc_func")
                        
                        if st.button("Calculate Derivative", key="calc_deriv_btn"):
                            try:
                                from sympy import symbols, sympify, diff, latex, simplify
                                from utils.validators import preprocess_function
                                
                                x_sym = symbols('x')
                                processed = preprocess_function(calc_func)
                                expr = sympify(processed)
                                
                                st.markdown("**f(x):**")
                                st.latex(f"{latex(expr)}")
                                
                                deriv = diff(expr, x_sym)
                                simplified = simplify(deriv)
                                
                                st.markdown("**f'(x):**")
                                st.latex(f"{latex(simplified)}")
                                
                                st.code(f"f'(x) = {simplified}", language="python")
                                
                            except Exception as e:
                                st.error(f"Error: {e}")


                # ============================================================================
                # PART 8: Additional helper - Derivative at specific points table
                # Can be added as an expander in any method that uses derivatives
                # ============================================================================

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
                            
                            # Curvature info would require f''
                            eval_data.append({
                                'x': f"{x_val:.6f}",
                                'f(x)': f"{f_val:.6f}",
                                "f'(x)": f"{fp_val:.6f}",
                                'Behavior': behavior
                            })
                        except:
                            pass
                    
                    st.dataframe(pd.DataFrame(eval_data), use_container_width=True, hide_index=True)


                # ============================================================================
                # INSTRUCTIONS FOR INTEGRATION
                # ============================================================================

                """
                TO INTEGRATE THIS CODE INTO YOUR APP.PY:

                1. NEWTON-RAPHSON (Part 1):
                - Find line ~850-900 where Newton-Raphson displays results
                - After "st.success(f'✅ {result['message']}')"
                - Insert PART 1 code

                2. SECANT METHOD (Part 2):
                - Find Secant method results section
                - After displaying the result message
                - Insert PART 2 code

                3. FIXED POINT METHOD (Part 3):
                - Find Fixed Point method results section
                - After displaying the result message
                - Insert PART 3 code

                4. HELPER FUNCTIONS (Part 4):
                - Add near the top after imports, before main app code
                - These can be called from anywhere

                5. COMPARISON MODE (Part 5):
                - Find "Compare All Methods" section
                - After the comparison table display
                - Insert PART 5 code

                6. DERIVATIVE ANALYSIS (Part 6):
                - In Newton-Raphson section
                - After the iteration table
                - Insert PART 6 code

                7. SIDEBAR TOOL (Part 7):
                - Optional: Add to sidebar for quick derivative calculations
                - Insert after method selection in sidebar

                8. EVALUATION TABLE (Part 8):
                - Helper function - add with other utility functions
                - Call it wherever you want to show derivative evaluations

                TESTING:
                - Test with: x**3 - x - 2
                - Test with: sin(x) - x/2
                - Test with: exp(x) - 3
                - Test with: x**2 - 4
                - Test with inverse trig: asin(x) - 0.5

                Each should show detailed differentiation steps!
                """

                # Metrics
                col1, col2, col3 = st.columns(3)
                root_val = result['root']
                
                col1.metric("Root", f"{root_val:.8f}" if number_format == "Decimal" else f"{root_val:.6e}")
                
                if method == "Fixed Point Method":
                    col2.metric("g(root)", f"{g(root_val):.6e}")
                else:
                    col2.metric("f(root)", f"{f(root_val):.2e}")
                
                col3.metric("Iterations", len(result['iterations']))
                
                st.markdown("---")
                
                # Iteration table
                st.subheader("📊 Iteration Table")
                df = pd.DataFrame(result['iterations'])
                
                # Create a display copy without altering original data
                df_display = df.copy()

                if number_format == "Scientific":
                    for col in df_display.select_dtypes(include=[np.number]).columns:
                        df_display[col] = df_display[col].map(lambda x: f"{x:.6e}")
                else:
                    for col in df_display.select_dtypes(include=[np.number]).columns:
                        df_display[col] = df_display[col].map(lambda x: f"{x:.10f}")

                st.dataframe(df_display, use_container_width=True, height=400)
                
                # Download
                csv = pd.DataFrame(result['iterations']).to_csv(index=False)
                st.download_button("📥 Download CSV", csv, f"{method.replace(' ', '_')}_results.csv", "text/csv")
                
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
                'x': [8.1, 8.3],
                'y': [16.94410, 17.56492]
            },
            2: {  # degree 2 → 3 points
                'x': [8.1, 8.3, 8.6],
                'y': [16.94410, 17.56492, 18.50515]
            },
            3: {  # degree 3 → 4 points (your required example)
                'x': [8.1, 8.3, 8.6, 8.7],
                'y': [16.94410, 17.56492, 18.50515, 18.82091]
            },
            4: {  # degree 4 → repeat last as placeholder, or add your own 5th value
                'x': [8.1, 8.3, 8.6, 8.7, 0],
                'y': [16.94410, 17.56492, 18.50515, 18.82091, 0]
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
                            st.code(f"f[x{i}, x{i+1}] = ({dd_table[i+1][0]:.6f} - {dd_table[i][0]:.6f}) / ({x_points_dd[i+1]:.4f} - {x_points_dd[i]:.4f}) = {result_val:.6f}")
                        
                        # Show second differences if applicable
                        if n > 2:
                            st.markdown("**Second Divided Differences:**")
                            for i in range(n - 2):
                                numerator = dd_table[i+1][1] - dd_table[i][1]
                                denominator = x_points_dd[i+2] - x_points_dd[i]
                                result_val = dd_table[i][2]
                                st.code(f"f[x{i}, x{i+1}, x{i+2}] = ({dd_table[i+1][1]:.6f} - {dd_table[i][1]:.6f}) / ({x_points_dd[i+2]:.4f} - {x_points_dd[i]:.4f}) = {result_val:.6f}")
                        
                        # Show third differences if applicable
                        if n > 3:
                            st.markdown("**Third Divided Differences:**")
                            for i in range(n - 3):
                                numerator = dd_table[i+1][2] - dd_table[i][2]
                                denominator = x_points_dd[i+3] - x_points_dd[i]
                                result_val = dd_table[i][3]
                                st.code(f"f[x{i}, x{i+1}, x{i+2}, x{i+3}] = ({dd_table[i+1][2]:.6f} - {dd_table[i][2]:.6f}) / ({x_points_dd[i+3]:.4f} - {x_points_dd[i]:.4f}) = {result_val:.6f}")
                    
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
                                'Value': f"{coeff:.6f}",
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
                                st.write(f"- Term {i+1}: `{coeff:.6f}` (constant term)")
                            else:
                                factors = " × ".join([f"(x - {result_dd['x_points'][j]:.4f})" for j in range(i)])
                                st.write(f"- Term {i+1}: `{coeff:.6f} × {factors}`")
                    
                    # Step 5: Evaluate
                    with st.expander(f"🎯 **Step 5: Evaluate at x = {eval_x_dd}**", expanded=True):
                        st.markdown(f"Let's evaluate P({eval_x_dd}) step by step using nested multiplication:")
                        
                        # Show Horner's method calculation
                        st.markdown("**Using Horner's Method (efficient evaluation):**")
                        
                        eval_steps = []
                        n = len(result_dd['coefficients'])
                        value = result_dd['coefficients'][n - 1]
                        
                        eval_steps.append(f"Start with last coefficient: {value:.6f}")
                        
                        for i in range(n - 2, -1, -1):
                            old_value = value
                            value = value * (eval_x_dd - result_dd['x_points'][i]) + result_dd['coefficients'][i]
                            eval_steps.append(f"Step {n-i}: {old_value:.6f} × ({eval_x_dd:.4f} - {result_dd['x_points'][i]:.4f}) + {result_dd['coefficients'][i]:.6f} = {value:.6f}")
                        
                        for step in eval_steps:
                            st.code(step)
                        
                        st.success(f"**Final Result:** P({eval_x_dd}) = {value:.6f}")
                
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
                        row = {'i': i, 'x_i': f"{x_points_dd[i]:.4f}", 'f[x_i]': f"{dd_table[i][0]:.6f}"}
                        
                        for j in range(1, n):
                            if i + j < n:
                                if j == 1:
                                    row[f'f[x_i,x_i+1]'] = f"{dd_table[i][j]:.6f}"
                                elif j == 2:
                                    row[f'f[x_i,...,x_i+2]'] = f"{dd_table[i][j]:.6f}"
                                else:
                                    row[f'f[x_i,...,x_i+{j}]'] = f"{dd_table[i][j]:.6f}"
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
                col2.metric(f"P({eval_x_dd})", f"{eval_result_dd:.6f}")
                col3.metric("Data Points Used", len(result_dd['points']))
                
                st.markdown("---")
                
                # Data points table
                st.markdown("### 📍 Input Data Points")
                points_df_dd = pd.DataFrame(result_dd['points'])
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

st.caption("Numerical Computing Project | Root Finding & Interpolation Calculator")
