import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from methods.bisection import bisection
from utils.validators import validate_function, validate_interval, preprocess_function

st.set_page_config(page_title="Root Finding Calculator", page_icon="📐", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Math-like font for function input */
    .stTextArea textarea {
        font-family: 'Courier New', 'Computer Modern', 'Times New Roman', serif;
        font-size: 18px !important;
        font-weight: 500;
        color: #1f4788;
        letter-spacing: 0.5px;
    }
    
    /* Better button styling */
    .stButton button {
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    /* Main title styling */
    h1 {
        color: #1f4788;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Better table styling */
    .dataframe {
        font-size: 14px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Success/Error boxes */
    .stSuccess {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    
    .stError {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #d1ecf1;
        border-left: 4px solid #0c5460;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📐 Numerical Root Finding Calculator")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Method selection badge
st.sidebar.markdown("### 🎯 Method Selection")
method = st.sidebar.selectbox("", ["Bisection Method"], label_visibility="collapsed")

# Function input with virtual keypad
st.sidebar.markdown("### 🔢 Function Builder")

# Initialize session state for function input
if 'func_input' not in st.session_state:
    st.session_state.func_input = "x**3 - x - 2"

# Display current function with math-like styling
func_str = st.sidebar.text_area(
    "Enter f(x):",
    value=st.session_state.func_input,
    height=100,
    help="Build your function using the keypad below or type directly",
    key="function_input"
)

# Update session state when text changes
st.session_state.func_input = func_str

# Virtual Keypad with better organization
st.sidebar.markdown("#### 🧮 Scientific Keypad")

# Row 1: Powers & Roots
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

# Row 2: Operators
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

# Row 3: Trigonometric
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

# Row 4: Advanced functions
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

# Row 5-8: Number pad
st.sidebar.markdown("##### Numbers & Brackets")
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

# Row 8: Zero and controls
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

# Quick examples
with st.sidebar.expander("📋 Quick Load Examples"):
    examples = {
        "x³ - x - 2": "x**3 - x - 2",
        "x² - 4": "x**2 - 4",
        "cos(x) - x": "cos(x) - x",
        "eˣ - 3": "exp(x) - 3",
        "sin(x) - x/2": "sin(x) - x/2",
        "ln(x) - 2": "log(x) - 2",
        "√x - 2": "sqrt(x) - 2",
        "x⁴ - 10": "x**4 - 10"
    }
    
    for name, formula in examples.items():
        if st.button(name, use_container_width=True, key=f"ex_{formula}"):
            st.session_state.func_input = formula
            st.rerun()

# Help message
with st.sidebar.expander("💡 Smart Input Tips"):
    st.markdown("""
    **Auto-multiplication works for:**
    - `2x` → `2*x`
    - `3cos(x)` → `3*cos(x)`
    - `(x+1)(x-1)` → `(x+1)*(x-1)`
    
    **Supported functions:**
    - Trig: `sin`, `cos`, `tan`
    - Exponential: `exp(x)` = eˣ
    - Logarithm: `log(x)` = ln(x)
    - Other: `sqrt`, `abs`, `pi`
    """)

st.sidebar.markdown("---")

# Validate function
is_valid, f, error_msg = validate_function(func_str)

if not is_valid:
    st.sidebar.error(f"❌ {error_msg}")
    st.error("⚠️ Please enter a valid function in the sidebar")
    st.stop()
else:
    st.sidebar.success("✅ Function is valid")
    
    # Show preprocessed version if different
    processed = preprocess_function(func_str)
    if processed != func_str:
        st.sidebar.info(f"📝 Interpreted as: `{processed}`")

# Parameters
st.sidebar.markdown("### 📏 Parameters")
col1, col2 = st.sidebar.columns(2)
a = col1.number_input("a (left):", value=-2.0, format="%.4f")
b = col2.number_input("b (right):", value=2.0, format="%.4f")
tolerance = st.sidebar.number_input("Tolerance (ε):", value=0.000001, format="%.1e", min_value=1e-12)
max_iter = st.sidebar.number_input("Max Iterations:", value=100, min_value=1, max_value=1000)

# Display options
st.sidebar.markdown("### 📊 Display Options")
show_detailed = st.sidebar.checkbox("📋 Show Detailed Solution", value=True)
show_graph = st.sidebar.checkbox("📈 Show Graphs", value=True)
number_format = st.sidebar.radio("Number Format:", ["Decimal", "Scientific"], horizontal=True)

st.sidebar.markdown("---")

# Calculate button
calculate = st.sidebar.button("🚀 Calculate Root", type="primary", use_container_width=True)

# Main content
if calculate:
    # Validate interval
    valid, msg = validate_interval(f, a, b)
    if not valid:
        st.error(f"❌ {msg}")
        st.stop()
    
    # Show detailed solution
    if show_detailed:
        st.subheader("📋 Detailed Solution")
        
        # IVT Test
        st.markdown("### 1️⃣ Intermediate Value Theorem (IVT) Test")
        fa = f(a)
        fb = f(b)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("f(a)", f"{fa:.6f}" if number_format == "Decimal" else f"{fa:.6e}")
        with col2:
            st.metric("f(b)", f"{fb:.6f}" if number_format == "Decimal" else f"{fb:.6e}")
        
        if fa * fb < 0:
            st.success("✅ **IVT Satisfied:** f(a) and f(b) have opposite signs → Root exists in [a, b]")
            st.latex(r"f(a) \cdot f(b) < 0 \quad \Rightarrow \quad \exists \, c \in [a,b] : f(c) = 0")
        else:
            st.error("❌ **IVT Not Satisfied:** f(a) and f(b) must have opposite signs")
            st.stop()
        
        # Formula for max iterations
        st.markdown("### 2️⃣ Theoretical Maximum Iterations")
        st.write("Formula for maximum iterations to achieve tolerance ε:")
        st.latex(r"n_{max} = \left\lceil \frac{\ln(b - a) - \ln(\epsilon)}{\ln(2)} \right\rceil")
        
        theoretical_max = np.ceil((np.log(b - a) - np.log(tolerance)) / np.log(2))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Given:**")
            st.write(f"• Interval: [{a}, {b}]")
            st.write(f"• Width: {b - a}")
            st.write(f"• Tolerance: {tolerance:.1e}")
        
        with col2:
            st.write("**Calculation:**")
            st.write(f"• ln({b - a}) = {np.log(b - a):.4f}")
            st.write(f"• ln({tolerance:.1e}) = {np.log(tolerance):.4f}")
            st.write(f"• Difference: {np.log(b - a) - np.log(tolerance):.4f}")
        
        with col3:
            st.write("**Result:**")
            st.write(f"• n_max / ln(2) = {(np.log(b - a) - np.log(tolerance)) / np.log(2):.2f}")
            st.metric("Maximum Iterations", f"{int(theoretical_max)}")
        
        # Error after n iterations formula
        st.markdown("### 3️⃣ Error Analysis After n Iterations")
        st.write("Error formula for Bisection Method:")
        st.latex(r"\text{Error}_n = \frac{b - a}{2^n}")
        
        st.write("**Error at different iterations:**")
        error_table_data = []
        for n in [1, 5, 10, 15, int(theoretical_max)]:
            error_n = (b - a) / (2**n)
            error_table_data.append({
                'n': n,
                'Formula': f"({b-a:.4f}) / 2^{n}",
                'Error': f"{error_n:.6e}" if number_format == "Scientific" else f"{error_n:.10f}",
                'Status': '✅ Converged' if error_n < tolerance else '⏳ Iterating'
            })
        
        error_df = pd.DataFrame(error_table_data)
        st.dataframe(error_df, use_container_width=True, hide_index=True)
        
        # Custom error calculation
        st.markdown("#### 🧮 Custom Error Calculator")
        col1, col2 = st.columns([1, 2])
        with col1:
            custom_n = st.number_input("Enter iteration (n):", min_value=1, max_value=200, value=10, key="custom_iter")
        
        custom_error = (b - a) / (2**custom_n)
        
        with col2:
            st.latex(rf"\text{{Error}}_{{{custom_n}}} = \frac{{{b - a:.4f}}}{{2^{{{custom_n}}}}} = {custom_error:.6e}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Iteration", custom_n)
        col2.metric("Error", f"{custom_error:.6e}" if number_format == "Scientific" else f"{custom_error:.10f}")
        col3.metric("Status", "✅ Converged" if custom_error < tolerance else "⏳ Not Converged")
        
        st.markdown("---")
    
    # Run bisection
    with st.spinner("🔄 Calculating root..."):
        result = bisection(f, a, b, tol=tolerance, max_iter=max_iter)
    
    # Show results
    if result['success']:
        st.success(f"✅ {result['message']}")
        
        # Metrics
        st.markdown("### 📊 Results Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        root_val = result['root']
        f_root = f(result['root'])
        actual_iter = len(result['iterations'])
        final_error = result['iterations'][-1]['error']
        
        with col1:
            st.metric("🎯 Root (x)", 
                     f"{root_val:.8f}" if number_format == "Decimal" else f"{root_val:.6e}")
        with col2:
            st.metric("📉 f(root)", 
                     f"{f_root:.2e}" if abs(f_root) < 0.001 else f"{f_root:.6f}")
        with col3:
            st.metric("🔄 Iterations", f"{actual_iter}")
        with col4:
            st.metric("📏 Final Error", 
                     f"{final_error:.6e}" if number_format == "Scientific" else f"{final_error:.10f}")
        
        st.markdown("---")
        
        # Iteration table
        st.subheader("📊 Iteration Table")
        df = pd.DataFrame(result['iterations'])
        
        # Format based on user preference
        if number_format == "Scientific":
            df_display = df.copy()
            df_display['a'] = df_display['a'].map('{:.6e}'.format)
            df_display['b'] = df_display['b'].map('{:.6e}'.format)
            df_display['c'] = df_display['c'].map('{:.6e}'.format)
            df_display['f(c)'] = df_display['f(c)'].map('{:.6e}'.format)
            df_display['error'] = df_display['error'].map('{:.6e}'.format)
        else:
            df_display = df.copy()
            df_display['a'] = df_display['a'].map('{:.10f}'.format)
            df_display['b'] = df_display['b'].map('{:.10f}'.format)
            df_display['c'] = df_display['c'].map('{:.10f}'.format)
            df_display['f(c)'] = df_display['f(c)'].map('{:.10f}'.format)
            df_display['error'] = df_display['error'].map('{:.10f}'.format)
        
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=f"bisection_{func_str[:20].replace('*','').replace(' ','_')}.csv",
            mime="text/csv",
            use_container_width=False
        )
        
        # Graphs
        if show_graph:
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Function Plot")
                
                # Create plot
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot function
                x_plot = np.linspace(a - 0.5, b + 0.5, 1000)
                try:
                    y_plot = f(x_plot)
                    ax.plot(x_plot, y_plot, 'b-', linewidth=2.5, label=f'f(x) = {func_str}')
                    
                    # Plot axes
                    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
                    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
                    
                    # Plot initial interval
                    ax.plot(a, f(a), 'go', markersize=12, label=f'a = {a}', zorder=5)
                    ax.plot(b, f(b), 'go', markersize=12, label=f'b = {b}', zorder=5)
                    
                    # Plot final root
                    root = result['root']
                    ax.plot(root, f(root), 'r*', markersize=20, label=f'Root = {root:.6f}', zorder=10)
                    ax.axvline(x=root, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
                    
                    ax.set_xlabel('x', fontsize=13, fontweight='bold')
                    ax.set_ylabel('f(x)', fontsize=13, fontweight='bold')
                    ax.set_title(f'Root of {func_str}', fontsize=14, fontweight='bold', pad=15)
                    ax.legend(loc='best', fontsize=10)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not plot: {e}")
            
            with col2:
                st.markdown("### 📉 Convergence Analysis")
                
                # Convergence plot
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                
                iterations_num = [it['n'] for it in result['iterations']]
                errors = [it['error'] for it in result['iterations']]
                
                ax2.semilogy(iterations_num, errors, 'b-o', linewidth=2.5, markersize=7, 
                           label='Actual Error', markerfacecolor='lightblue', markeredgecolor='blue', markeredgewidth=2)
                ax2.axhline(y=tolerance, color='red', linestyle='--', linewidth=2, 
                          label=f'Tolerance = {tolerance:.1e}', alpha=0.8)
                
                # Theoretical error line
                theoretical_errors = [(b - a) / (2**n) for n in iterations_num]
                ax2.semilogy(iterations_num, theoretical_errors, 'g--', linewidth=2, 
                           alpha=0.6, label='Theoretical Error')
                
                ax2.set_xlabel('Iteration (n)', fontsize=13, fontweight='bold')
                ax2.set_ylabel('Error (log scale)', fontsize=13, fontweight='bold')
                ax2.set_title('Error vs Iteration', fontsize=14, fontweight='bold', pad=15)
                ax2.grid(True, alpha=0.3, which='both', linestyle='--')
                ax2.legend(fontsize=10, loc='upper right')
                
                st.pyplot(fig2)
    
    else:
        st.error(f"❌ {result['message']}")

else:
    # Welcome screen
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

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Numerical Computing Project</strong> • Built with ❤️ using Streamlit</p>
    <p style='font-size: 12px;'>Bisection Method • Root Finding Calculator • Interactive Learning Tool</p>
</div>
""", unsafe_allow_html=True)