# 📘 Numerical Methods Solver – Streamlit App

A fully interactive Streamlit-based application that solves nonlinear equations using multiple numerical root-finding algorithms.  
Includes plotting, iteration tables, keypad input, method-specific parameters, and a comparison mode.

---

## 🚀 Features

### 🔢 1. Multiple Numerical Methods
- Bisection Method  
- False Position Method  
- Newton–Raphson Method  
- Secant Method  
- Fixed Point Iteration (with g(x))  
- Compare All Methods Mode  

Each method includes:
- Input validation  
- Iteration table  
- Error computation  
- Convergence detection  
- Method-specific parameter controls  

---

### 🎛️ 2. Smart Sidebar Inputs

| Method | Interval `[a, b]` | Initial Guess | g(x) | Notes |
|--------|---------------------|----------------|------|-------|
| Bisection | ✔️ | ❌ | ❌ | Requires f(a)·f(b) < 0 |
| False Position | ✔️ | ❌ | ❌ | Bracketing required |
| Newton–Raphson | ❌ | ✔️ (x₀) | ❌ | Uses derivative |
| Secant | ❌ | ✔️ (x₀, x₁) | ❌ | Two initial guesses |
| Fixed Point | ✔️ | ✔️ | ✔️ | x = g(x) iteration |
| Compare All | ✔️ | ✔️ | ✔️ | Runs all methods |

---

### 🧠 3. Expression Parser & Validator
Supports:
- Mathematical functions: `sin`, `cos`, `log`, `exp`, `sqrt`, etc.  
- Constants: `pi`, `e`  
- User-defined function `f(x)`  
- User-defined `g(x)` for Fixed Point  
- Automatic derivative computation (Newton–Raphson)  

Invalid expressions generate real-time warnings.

---

## 📊 4. Visualization Tools
- Function plot  
- Iteration movement plot  
- g(x) vs x graph (Fixed Point)  
- Adjustable intervals for visualization  
- Iteration table containing:
  - Approximated root  
  - f(x)  
  - Error  
  - Iteration number  

---

## 🖩 5. Virtual Scientific Keypad
A clickable keypad to enter:
- Numbers  
- Operators  
- Functions  
- Constants (π, e)  
- Parentheses  

Reduces typing errors and helps beginners.

---

## 🔬 6. Compare-All Mode
Runs **all numerical methods side-by-side**, showing:
- Individual iteration tables  
- Convergence summary  
- Final outputs  
- Execution time  
- Combined comparison graph  

Useful for analysis and lab reports.

---

## 🛠️ Tech Stack

| Component | Purpose |
|----------|----------|
| Python | Core language |
| Streamlit | Frontend/UI |
| SymPy | Parser, differentiation |
| NumPy | Numerical operations |
| Matplotlib | Plotting |
| Pandas | Iteration tables |

---

## 📂 Folder Structure

project/
│── app.py # Main Streamlit app
│── requirements.txt # Dependencies
│── methods/
│ ├── bisection.py
│ ├── false_position.py
│ ├── secant.py
│ ├── newton.py
│ ├── fixed_point.py
│── utils/
│ ├── parser.py
│ ├── keypad.py
│ ├── plotting.py
│ ├── tables.py
│── README.md

yaml
Copy code

---

## ▶️ Running the App

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Run the app
bash
Copy code
streamlit run app.py
The app will open automatically in your browser.

🎯 Typical Workflow
Select a numerical method

Enter f(x)

Provide required parameters

Validate inputs

Click Solve

View:

Root

Iteration table

Graphs

Optionally choose Compare All

📚 Educational Purpose
Ideal for:

Numerical Computing labs

DSA / Mathematical Computing courses

University assignments

Demonstrating convergence visually

🤝 Contributions
Pull requests are welcome!
Suggestions for new methods (Müller, Steffensen, Hybrid, etc.) are appreciated.
