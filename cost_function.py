from qiskit_optimization import QuadraticProgram
import numpy as np

def get_profit_coefficients_vectorized(df, gp, w, wij, alpha, beta, kappa, fric, interactions):
    M = len(df)
    
    # Extract features as NumPy arrays for clean broadcasting
    N = df["n_employees"].to_numpy()
    C_dep = df["cloud_dep"].to_numpy()
    R = df["remote_ratio"].to_numpy()
    As = df["attack_surface"].to_numpy()
    I = df["past_incidents"].to_numpy()
    g = df["industry_risk"].to_numpy()
    Rev = df["rev"].to_numpy()
    pcomp = df["competitor_price"].to_numpy()

    # Base Risk & Intensity
    A_hat = gp["a0"] + gp["a1"] * N
    E_hat = gp["e0"] + gp["e1"] * (As / 100.0) + gp["e2"] * C_dep + gp["e3"] * R + gp["e4"] * np.minimum(I, gp["Imax"])
    Lambda = gp["lambda0"] * g * (1 + gp["lambda_C"] * C_dep) * (1 + gp["lambda_R"] * R) * A_hat * E_hat

    # Security Score Linearization (Scalars)
    S0 = 1.0 / (1.0 + np.exp(-gp["z0"]))
    S0_deriv = S0 * (1.0 - S0)
    c0 = 1.0 - S0

    s_i = S0_deriv * w
    s_ij = np.zeros((6, 6))
    for (i, j), w_val in wij.items():
        s_ij[i, j] = S0_deriv * w_val

    # Ransomware Probability Coefficients (Scalars)
    alpha_ij = np.zeros((6, 6))
    beta_ij = np.zeros((6, 6))
    for (i, j), (a_val, b_val) in interactions.items():
        alpha_ij[i, j] = a_val
        beta_ij[i, j] = b_val

    L = np.zeros(6)
    for i in range(6):
        L[i] = -(1 - S0) * alpha[i] - S0_deriv * w[i] * (1 - alpha[i])

    Q = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1, 6):
            Q[i, j] = -S0_deriv * wij.get((i, j), 0) - (1 - S0) * alpha_ij[i, j] \
                      + alpha[i] * S0_deriv * w[j] + alpha[j] * S0_deriv * w[i]

    # Arrays with shape (M,) mapped to (M, 6) or (M, 6, 6)
    P0 = Lambda * c0
    P_i = Lambda[:, np.newaxis] * L
    P_ij = Lambda[:, np.newaxis, np.newaxis] * Q

    # Impact Linearization
    Rev_daily = Rev / 365.0
    I0_raw = gp["c1"] * A_hat + Rev_daily * (gp["d0"] + gp["d1"] * A_hat * (1 - S0)) + gp["gamma"] * Rev

    I_raw = np.zeros((M, 6))
    for i in range(6):
        I_raw[:, i] = -Rev_daily * gp["d1"] * A_hat * s_i[i]

    I_ij_raw = np.zeros((M, 6, 6))
    for i in range(6):
        for j in range(i + 1, 6):
            I_ij_raw[:, i, j] = -Rev_daily * gp["d1"] * A_hat * s_ij[i, j]

    I_linear = np.zeros((M, 6))
    for i in range(6):
        I_linear[:, i] = I_raw[:, i] - I0_raw * beta[i] + I_raw[:, i] * beta[i]

    I_quad = np.zeros((M, 6, 6))
    for i in range(6):
        for j in range(i + 1, 6):
            I_quad[:, i, j] = I_ij_raw[:, i, j] - I0_raw * beta_ij[i, j] + I_raw[:, i] * beta[j] + I_raw[:, j] * beta[i]

    # Expected Payout
    d = gp["deductible"]
    u = gp["limit"]
    I0_tilde = I0_raw - d

    E0 = P0 * I0_tilde
    E_linear = np.zeros((M, 6))
    for i in range(6):
        E_linear[:, i] = P0 * I_linear[:, i] + I0_tilde * P_i[:, i] + P_i[:, i] * I_linear[:, i]

    E_quad = np.zeros((M, 6, 6))
    for i in range(6):
        for j in range(i + 1, 6):
            E_quad[:, i, j] = P0 * I_quad[:, i, j] + I0_tilde * P_ij[:, i, j] + P_i[:, i] * I_linear[:, j] + P_i[:, j] * I_linear[:, i]

    # Acceptance Function Linearization
    eta_tilde_0 = gp["eta0"] + gp["eta_p"] * pcomp + gp["eta_u"] * u - gp["eta_d"] * d
    L0 = gp["eta0"] + gp["eta_u"] * u - gp["eta_d"] * d
    A0 = 1.0 / (1.0 + np.exp(-L0))

    alpha0 = A0 + A0 * (1 - A0) * (eta_tilde_0 - L0)
    alpha_p = -A0 * (1 - A0) * gp["eta_p"]
    alpha_i = -A0 * (1 - A0) * gp["eta_c"] * fric

    # Final Profit Coefficients
    C_const = -alpha0 * E0
    C_p = alpha0 - alpha_p * E0
    C_pp = np.full(M, alpha_p) # Broadcasting scalar to length M for consistency

    C_i = np.zeros((M, 6))
    for i in range(6):
        C_i[:, i] = -alpha0 * (kappa[i] + E_linear[:, i]) - E0 * alpha_i[i] - alpha_i[i] * (kappa[i] + E_linear[:, i])

    C_ip = np.zeros((M, 6))
    for i in range(6):
        C_ip[:, i] = alpha_i[i] - alpha_p * (kappa[i] + E_linear[:, i])

    C_ij = np.zeros((M, 6, 6))
    for i in range(6):
        for j in range(i + 1, 6):
            C_ij[:, i, j] = -alpha0 * E_quad[:, i, j] - alpha_i[i] * (kappa[j] + E_linear[:, j]) - alpha_i[j] * (kappa[i] + E_linear[:, i])

    return {
        "C": C_const,       # Shape: (M,)
        "Cp": C_p,          # Shape: (M,)
        "Cpp": C_pp,        # Shape: (M,)
        "Ci": C_i,          # Shape: (M, 6)
        "Cip": C_ip,        # Shape: (M, 6)
        "Cij": C_ij         # Shape: (M, 6, 6)
    }


def build_qiskit_quadratic_program(coefficients, p_min=500, p_max=6000):
    """
    Transforms the linearized profit coefficients into a Qiskit QuadraticProgram.
    """
    qp = QuadraticProgram(name="Ransomware_Premium_Optimization")
    
    # 1. Define Variables
    control_names = ["x_MFA", "x_EDR", "x_Offline_Backup", "x_Patch_Mgmt", "x_IR_Plan", "x_Net_Seg"]
    for name in control_names:
        qp.binary_var(name=name)
        
    # The document expresses p as continuous in the quadratic form. 
    qp.integer_var(name="p", lowerbound=int(p_min), upperbound=int(p_max))    
    # Extract coefficient matrices
    C_const = coefficients["C"]
    C_p = coefficients["Cp"]
    C_pp = coefficients["Cpp"]
    C_i = coefficients["Ci"]
    C_ip = coefficients["Cip"]
    C_ij = coefficients["Cij"]
    
    # 2. Build Linear Objective Terms
    linear_obj = {"p": C_p}
    for idx, name in enumerate(control_names):
        if C_i[idx] != 0:
            linear_obj[name] = C_i[idx]
            
    # 3. Build Quadratic Objective Terms
    quadratic_obj = {("p", "p"): C_pp}
    
    for i in range(6):
        # Interaction between control i and premium p
        if C_ip[i] != 0:
            quadratic_obj[(control_names[i], "p")] = C_ip[i]
            
        # Interaction between control i and control j
        for j in range(i + 1, 6):
            if C_ij[i, j] != 0:
                quadratic_obj[(control_names[i], control_names[j])] = C_ij[i, j]
                
    # 4. Set Objective
    # The document evaluates Expected Profit, requiring maximization.
    qp.maximize(constant=C_const, linear=linear_obj, quadratic=quadratic_obj)
    
    return qp

def generate_qps_for_dataset(batched_coeffs, num_companies, p_min=5000, p_max=60000):
    """
    Iterates through the batched coefficients and generates a list of 
    Qiskit QuadraticProgram objects, one for each company.
    """
    qps = []
    
    for k in range(num_companies):
        # Extract the specific scalars and arrays for the k-th company
        single_coeffs = {
            "C": batched_coeffs["C"][k],
            "Cp": batched_coeffs["Cp"][k],
            "Cpp": batched_coeffs["Cpp"][k],
            "Ci": batched_coeffs["Ci"][k],      # Shape: (6,)
            "Cip": batched_coeffs["Cip"][k],    # Shape: (6,)
            "Cij": batched_coeffs["Cij"][k]     # Shape: (6, 6)
        }
        
        # Build the QP for this specific company
        qp = build_qiskit_quadratic_program(single_coeffs, p_min, p_max)
        
        # Optional: Name the QP uniquely for tracking
        qp.name = f"Ransomware_Premium_Optimization_Company_{k}"
        
        qps.append(qp)
        
    return qps

