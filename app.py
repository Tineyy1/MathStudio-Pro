import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import plotly.graph_objects as go
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="MathStudio Pro", page_icon="📐", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📐 MathStudio Pro")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Select Module", ["Root Finding Analysis", "Advanced Matrix Operations"])
st.sidebar.markdown("---")
st.sidebar.info("Developed with Streamlit & Python")

# ==========================================
# MODULE 1: ROOT FINDING
# ==========================================
if app_mode == "Root Finding Analysis":
    st.title("Root Finding Analysis")
    st.markdown("Analyze equations and find roots using numerical methods with interactive visualizations.")
    
    col_input, col_results = st.columns([1, 2.5])
    
    with col_input:
        st.subheader("Parameters")
        eq_str = st.text_input("Equation f(x)", value="x**3 - x - 2")
        method = st.selectbox("Algorithm", ["Incremental Search", "Bisection Method", "Regula-Falsi", "Newton-Raphson", "Secant Method"])
        
        if method in ["Bisection Method", "Regula-Falsi", "Incremental Search"]:
            xl = st.number_input("Lower Bound (xl)", value=1.0)
            xu = st.number_input("Upper Bound (xu)", value=2.0)
        elif method == "Newton-Raphson":
            x0 = st.number_input("Initial Guess (x0)", value=1.0)
        elif method == "Secant Method":
            x0 = st.number_input("First Guess (x0)", value=1.0)
            x1 = st.number_input("Second Guess (x1)", value=2.0)
            
        tol = st.number_input("Tolerance", value=0.0001, format="%.5f")
        max_iter = st.number_input("Max Iterations", value=50, step=1)
        solve_btn = st.button("Calculate Root")

    with col_results:
        if solve_btn:
            try:
                x = sp.Symbol('x')
                expr = sp.sympify(eq_str)
                f = sp.lambdify(x, expr, 'numpy')
                df = sp.lambdify(x, sp.diff(expr, x), 'numpy')

                results, root, iterations, final_err = [], None, 0, 0
                
                # Algorithms (Logic remains the same, optimized for metric tracking)
                if method == "Bisection Method":
                    for i in range(max_iter):
                        xr = (xl + xu) / 2
                        err = abs(xu - xl) / 2
                        results.append({"Iter": i+1, "xl": xl, "xu": xu, "xr": xr, "f(xr)": f(xr), "Error": err})
                        if f(xr) == 0 or err < tol:
                            root, iterations, final_err = xr, i+1, err
                            break
                        if f(xl) * f(xr) < 0: xu = xr
                        else: xl = xr

                elif method == "Regula-Falsi":
                    for i in range(max_iter):
                        xr = xu - (f(xu)*(xl - xu)) / (f(xl) - f(xu))
                        err = abs(f(xr))
                        results.append({"Iter": i+1, "xl": xl, "xu": xu, "xr": xr, "f(xr)": f(xr), "Error": err})
                        if err < tol:
                            root, iterations, final_err = xr, i+1, err
                            break
                        if f(xl) * f(xr) < 0: xu = xr
                        else: xl = xr

                elif method == "Newton-Raphson":
                    xr = x0
                    for i in range(max_iter):
                        fxr, dfxr = f(xr), df(xr)
                        xr_new = xr - fxr/dfxr
                        err = abs(xr_new - xr)
                        results.append({"Iter": i+1, "xi": xr, "f(xi)": fxr, "f'(xi)": dfxr, "xi+1": xr_new, "Error": err})
                        xr = xr_new
                        if err < tol:
                            root, iterations, final_err = xr, i+1, err
                            break

                elif method == "Secant Method":
                    for i in range(max_iter):
                        fx1, fx0 = f(x1), f(x0)
                        x2 = x1 - (fx1 * (x0 - x1)) / (fx0 - fx1)
                        err = abs(x2 - x1)
                        results.append({"Iter": i+1, "x(i-1)": x0, "x(i)": x1, "x(i+1)": x2, "f(x(i+1))": f(x2), "Error": err})
                        x0, x1 = x1, x2
                        if err < tol:
                            root, iterations, final_err = x2, i+1, err
                            break
                            
                elif method == "Incremental Search":
                    step, curr_x = 0.1, xl
                    for i in range(max_iter):
                        next_x = curr_x + step
                        results.append({"Iter": i+1, "x": curr_x, "f(x)": f(curr_x)})
                        if f(curr_x) * f(next_x) < 0:
                            root, iterations = (curr_x + next_x)/2, i+2
                            break
                        curr_x = next_x

                if root is not None:
                    st.toast('Calculation Complete!', icon='✅')
                    
                    # --- METRIC CARDS ---
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Calculated Root", f"{root:.6f}")
                    m2.metric("Total Iterations", iterations)
                    m3.metric("Final Error", f"{final_err:.6f}" if final_err else "N/A")
                    st.divider()

                    # --- INTERACTIVE PLOTLY GRAPH ---
                    x_vals = np.linspace(root - 3, root + 3, 300)
                    y_vals = f(x_vals)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name='f(x)', line=dict(color='royalblue', width=2)))
                    fig.add_hline(y=0, line_dash="dash", line_color="black")
                    fig.add_vline(x=0, line_dash="dash", line_color="black")
                    fig.add_trace(go.Scatter(x=[root], y=[0], mode='markers', name='Root', marker=dict(color='red', size=12, symbol='x')))
                    
                    fig.update_layout(title="Interactive Function Graph", xaxis_title="X Axis", yaxis_title="Y Axis", hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                    # --- EXPANDABLE TABLE ---
                    with st.expander("📊 View Detailed Iteration History"):
                        st.dataframe(pd.DataFrame(results), use_container_width=True)

            except Exception as e:
                st.error(f"Error evaluating equation: Make sure it is valid Python math (e.g., use ** for exponents). Details: {e}")

# ==========================================
# MODULE 2: MATRIX OPERATIONS
# ==========================================
elif app_mode == "Advanced Matrix Operations":
    st.title("Advanced Matrix Operations")
    st.markdown("Use the interactive spreadsheets below to input your matrix data.")
    
    op = st.selectbox("Select Operation", ["Addition", "Multiplication", "System of Equations (Ax = B)", "Adjoint", "Inverse", "Determinant", "Power of Matrix", "Transpose"])
    
    st.divider()
    
    # Dynamic Layout based on Operation
    if op in ["Addition", "Multiplication", "System of Equations (Ax = B)"]:
        col1, col2 = st.columns(2)
    else:
        col1, col2 = st.columns([1, 1]) # Keep it centered
        
    with col1:
        st.subheader("Matrix A")
        c1, c2 = st.columns(2)
        rows_A = c1.number_input("Rows A", 1, 10, 3)
        cols_A = c2.number_input("Cols A", 1, 10, 3)
        
        # INTERACTIVE DATA EDITOR FOR A
        df_A = pd.DataFrame(np.zeros((rows_A, cols_A)), columns=[f"Col {i+1}" for i in range(cols_A)])
        edited_A = st.data_editor(df_A, use_container_width=True, key="matrix_a")
        A = edited_A.to_numpy()

    if op in ["Addition", "Multiplication", "System of Equations (Ax = B)"]:
        with col2:
            st.subheader("Matrix B")
            if op == "System of Equations (Ax = B)":
                st.info("Matrix B must be a single column (Results vector)")
                rows_B = rows_A
                cols_B = 1
            elif op == "Addition":
                rows_B, cols_B = rows_A, cols_A
            else: # Multiplication
                c1, c2 = st.columns(2)
                rows_B = c1.number_input("Rows B", 1, 10, cols_A, disabled=True)
                cols_B = c2.number_input("Cols B", 1, 10, 3)
                
            # INTERACTIVE DATA EDITOR FOR B
            df_B = pd.DataFrame(np.zeros((rows_B, cols_B)), columns=[f"Col {i+1}" for i in range(cols_B)])
            edited_B = st.data_editor(df_B, use_container_width=True, key="matrix_b")
            B = edited_B.to_numpy()
            
    if op == "Power of Matrix":
        with col2:
            st.subheader("Settings")
            power = st.number_input("Calculate to the power of (n):", value=2, step=1)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Execute Matrix Operation", use_container_width=True):
        try:
            with st.spinner("Calculating..."):
                time.sleep(0.5) # Slight delay for professional UX feel
                st.subheader("Resulting Matrix")
                
                if op == "Addition":
                    result = A + B
                elif op == "Multiplication":
                    result = np.matmul(A, B)
                elif op == "Transpose":
                    result = A.T
                elif op == "Determinant":
                    result = np.linalg.det(A)
                    st.metric("Determinant Value", f"{result:.4f}")
                    result = None # Skip table
                elif op == "Inverse":
                    result = np.linalg.inv(A)
                elif op == "Adjoint":
                    result = np.linalg.inv(A) * np.linalg.det(A)
                    result = np.round(result, 4)
                elif op == "Power of Matrix":
                    result = np.linalg.matrix_power(A, power)
                elif op == "System of Equations (Ax = B)":
                    result = np.linalg.solve(A, B)
                    st.success("Solutions found for Vector X:")
                    
                # Display output beautifully
                if result is not None:
                    st.dataframe(pd.DataFrame(result), use_container_width=True)
                    st.toast('Operation Successful!', icon='✅')
                    
        except np.linalg.LinAlgError as e:
            st.error(f"Mathematical Error: {e} (e.g., Matrix might be singular/non-invertible)")
        except ValueError as e:
            st.error(f"Dimension Error: {e}")