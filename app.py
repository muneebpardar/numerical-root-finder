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
from utils.validators import validate_function, validate_interval, preprocess_function

st.set_page_config(
    page_title="Numerical Methods Calculator",
    page_icon="🔢",
    layout="wide"
)


st.markdown("""
    <style>
    .force-header h1 {
    color: white !important;
    -webkit-text-fill-color: white !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: block !important;
    position: relative;
    z-index: 9999;
    }
    /* Function input styling */
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        font-size: 18px !important;
        font-weight: 600;
        color: #1f4788;
        letter-spacing: 0.5px;
        background-color: #f8f9fa;
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 10px;
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
        color: white;
        font-size: 16px;
        font-weight: 700;
        border: none;
    }
    
    /* Title styling */
    h1 {
        color: #1f4788;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #2c3e50;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 600;
        color: #7f8c8d;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 13px;
    }
    
    .dataframe thead tr th {
        background-color: #667eea !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
    }
    
    /* Error boxes */
    .stError {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #d1ecf1;
        border-left: 5px solid #0c5460;
        padding: 15px;
        border-radius: 5px;
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Enhanced Radio buttons */
    .stRadio > div {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    /* Download button enhancement */
    .stDownloadButton button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
        color: white !important;
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
    ["🎯 Root Finding", "📊 Lagrange Interpolation", "🔢 Divided Difference Interpolation"],
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

    # Advanced
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

    if not is_valid:
        st.sidebar.error(f"❌ Invalid Function")
        st.error(f"⚠️ **Error:** {error_msg}")
        st.info("💡 **Tip:** Use the keypad buttons or check syntax. Examples: x**2, sin(x), exp(x)")
        st.stop()
    else:
        st.sidebar.success("✅ Valid function")
        
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
                    terms_for_sum.append(f"({yi:.6f} × [{numerator_str}] / {denominator:.6f})")
            
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
                                        poly_terms.append(f"+ {coeff:.6f}")
                                    else:
                                        poly_terms.append(f"{coeff:.6f}")
                                elif power == 1:
                                    if coeff > 0 and poly_terms:
                                        poly_terms.append(f"+ {coeff:.6f}x")
                                    elif coeff < 0:
                                        poly_terms.append(f"- {abs(coeff):.6f}x")
                                    else:
                                        poly_terms.append(f"{coeff:.6f}x")
                                else:
                                    if coeff > 0 and poly_terms:
                                        poly_terms.append(f"+ {coeff:.6f}x^{power}")
                                    elif coeff < 0:
                                        poly_terms.append(f"- {abs(coeff):.6f}x^{power}")
                                    else:
                                        poly_terms.append(f"{coeff:.6f}x^{power}")
                            
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
                            'Result': f"{term_val:.6f}"
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
st.caption("Numerical Computing Project | Root Finding & Interpolation Calculator")
