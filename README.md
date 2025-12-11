# 📘 Numerical Methods Solver – Streamlit App

An interactive Streamlit application that provides multiple numerical algorithms for solving nonlinear equations and performing interpolation.  
Includes root-finding methods, Lagrange interpolation, plotting, iteration tables, keypad input, convergence validation, and a comparison mode.

---

## 🚀 Features

### 🔢 1. Numerical Methods Included
#### **Root-Finding Methods**
- Bisection Method  
- False Position Method  
- Newton–Raphson Method  
- Secant Method  
- Fixed Point Iteration (with g(x))  
- Compare All Methods Mode  

#### **Interpolation**
- **Lagrange Interpolation Method**  
  - Enter data points  
  - Generates the Lagrange polynomial  
  - Plots interpolation curve and given points  
  - Supports evaluation at any value of x  

---

## 🎛️ 2. Smart Sidebar Inputs

### **Root-Finding Parameter Matrix**
| Method | Interval `[a, b]` | Initial Guess | g(x) | Notes |
|--------|---------------------|----------------|------|-------|
| Bisection | ✔️ | ❌ | ❌ | f(a)·f(b) < 0 required |
| False Position | ✔️ | ❌ | ❌ | Bracketing required |
| Newton–Raphson | ❌ | ✔️ (x₀) | ❌ | Uses derivative |
| Secant | ❌ | ✔️ (x₀, x₁) | ❌ | Two initial guesses |
| Fixed Point | ✔️ | ✔️ | ✔️ | x = g(x) |
| Compare All | ✔️ | ✔️ | ✔️ | Runs all |
| Lagrange | ❌ | ❌ | ❌ | Requires data points |

### **Lagrange Inputs**
- Number of data points  
- x-values list  
- y-values list  
- Evaluation point (optional)

---

## 🧠 3. Expression Parser & Validator
Works for:
- Functions like `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`  
- Constants `pi`, `e`  
- User-defined `f(x)`  
- User-defined `g(x)`  
- Automatic derivative computation for Newton–Raphson  

Invalid expressions show instant warnings.

---

## 📊 4. Visualization Tools
### **Root-Finding**
- Function plot  
- Iteration movement graph  
- g(x) vs x plot for Fixed Point  
- Iteration table (root, f(x), error, iteration number)

### **Lagrange Interpolation**
- Interpolated polynomial plot  
- Visual markers for given data points  
- Evaluation of polynomial at user-input x  
- Display of full symbolic polynomial  

---

## 🖩 5. Virtual Scientific Keypad
Includes:
- Digits  
- Operators  
- Functions  
- Constants (π, e)  
- Parentheses  

Reduces typing mistakes.

---

## 🔬 6. Compare-All Mode
Runs every root-finding method side-by-side and shows:
- Root  
- Iteration tables  
- Convergence speed  
- Final error  
- Execution time  
- Combined comparison plot  

Excellent for assignments and analysis.

---

## 🛠️ Tech Stack

| Component | Purpose |
|----------|----------|
| Python | Core logic |
| Streamlit | UI frontend |
| SymPy | Parsing, differentiation, symbolic interpolation |
| NumPy | Numerical operations |
| Matplotlib | Plots |
| Pandas | Tables |

---

## 📂 Folder Structure

project/
│── app.py # Main Streamlit app
│── requirements.txt
│── methods/
│ ├── bisection.py
│ ├── false_position.py
│ ├── secant.py
│ ├── newton.py
│ ├── fixed_point.py
│ ├── lagrange.py # NEW: interpolation logic
│── utils/
│ ├── parser.py
│ ├── keypad.py
│ ├── plotting.py
│ ├── tables.py
│── README.md

---

## ▶️ Running the App

### Install dependencies
```bash
pip install -r requirements.txt
Run the app
bash
Copy code
streamlit run app.py
```
---
### 🎯 Typical Workflow
## Root-Finding
Select a method

Enter f(x)

Provide parameters (interval or guesses)

Click Solve

Review tables and plots

---
## Lagrange Interpolation
Enter data points

Generate polynomial

Plot interpolation

(Optional) Evaluate at specific x

---
## 📚 Educational Purpose
Perfect for:

Numerical Computing labs

Mathematical Computing courses

Engineering/problem-solving demonstrations

Visualizing convergence and interpolation

---
## 🤝 Contributions
Pull requests are welcome!
Additional methods (Müller, Hermite, Newton Interpolation, Gauss methods, etc.) are encouraged.




