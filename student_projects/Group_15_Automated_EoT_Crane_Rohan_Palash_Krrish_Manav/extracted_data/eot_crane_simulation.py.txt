# AI-assisted: Gemini used for translating dynamic EOMs into solve_ivp simulation syntax and generating matplotlib validation plots.

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------------------------
# 1. System Parameters
# ---------------------------------
Mb = 500.0   # Mass of bridge (kg)
mt = 100.0   # Mass of trolley (kg)
mp = 50.0    # Mass of payload (kg)
g = 9.81     # Gravity (m/s^2)
L_base = 10.0 # Standard Hoist cable length (m) for baseline test

# ---------------------------------
# 2. Dynamic Equations of Motion
# ---------------------------------
# State vector Y = [x, x_dot, theta_x, theta_x_dot]
def crane_dynamics(t, Y, L, cx=0.0):
    x, x_dot, theta, theta_dot = Y
    
    # Motor inputs (0 for free-response validation test)
    Fx = 0.0 
    
    # Solving the coupled equations derived in Phase 2 for accelerations:
    # 1. (Mb + mp)*x_ddot + mp*L*theta_ddot = Fx - cx*x_dot
    # 2. x_ddot + L*theta_ddot = -g*theta
    # Substituting (2) into (1) gives:
    x_ddot = (Fx - cx*x_dot + mp*g*theta) / Mb
    theta_ddot = (-g*theta - x_ddot) / L
    
    return [x_dot, x_ddot, theta_dot, theta_ddot]

# ---------------------------------
# 3. Initial Conditions & Setup
# ---------------------------------
# Bridge at 0m, stationary. Payload at 0.17 rad (~10 degrees swing), stationary.
Y0 = [0.0, 0.0, 0.17, 0.0]
t_span = (0, 20)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# ---------------------------------
# 4. Run Baseline Simulation
# ---------------------------------
sol_base = solve_ivp(
    fun=lambda t, y: crane_dynamics(t, y, L_base), 
    t_span=t_span, 
    y0=Y0, 
    t_eval=t_eval, 
    method='RK45'
)

x_dot_vals = sol_base.y[1]
theta_vals = sol_base.y[2]
theta_dot_vals = sol_base.y[3]

# ---------------------------------
# 5. Energy Calculations (CO5 Validation)
# ---------------------------------
# Kinetic Energy (T): Bridge translation + Payload absolute velocity
# Vp^2 ≈ (x_dot + L*theta_dot)^2 for small angles
Vp_sq = (x_dot_vals + L_base * theta_dot_vals)**2
T_sys = 0.5 * Mb * x_dot_vals**2 + 0.5 * mp * Vp_sq

# Potential Energy (V): Relative to the lowest hanging point
# h = L - L*cos(theta)
h = L_base - L_base * np.cos(theta_vals)
V_sys = mp * g * h

# Total Energy (E)
E_sys = T_sys + V_sys

# ---------------------------------
# 6. Run Sensitivity Analysis (Varying L)
# ---------------------------------
L_short = 5.0
sol_short = solve_ivp(
    fun=lambda t, y: crane_dynamics(t, y, L_short), 
    t_span=t_span, y0=Y0, t_eval=t_eval, method='RK45'
)

L_long = 15.0
sol_long = solve_ivp(
    fun=lambda t, y: crane_dynamics(t, y, L_long), 
    t_span=t_span, y0=Y0, t_eval=t_eval, method='RK45'
)

# ---------------------------------
# 7. Plotting the Validation Evidence
# ---------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Plot 1: Time Response
ax1.plot(sol_base.t, theta_vals, label=f'Swing Angle (L={L_base}m)', color='blue', linewidth=2)
ax1.set_title('Time Response: Payload Swing (Free Response Validation)')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Amplitude [rad]')
ax1.grid(True)
ax1.legend()

# Plot 2: Energy Conservation (CO5)
ax2.plot(sol_base.t, T_sys, label='Kinetic Energy (T)', color='green')
ax2.plot(sol_base.t, V_sys, label='Potential Energy (V)', color='orange')
ax2.plot(sol_base.t, E_sys, label='Total Energy (E)', color='red', linestyle='--', linewidth=2)
ax2.set_title('Energy vs Time (Conservation Check)')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Energy [Joules]')
ax2.grid(True)
ax2.legend()

# Plot 3: Sensitivity Analysis
ax3.plot(sol_short.t, sol_short.y[2], label=f'Short Cable (L={L_short}m)', color='purple')
ax3.plot(sol_base.t, sol_base.y[2], label=f'Base Cable (L={L_base}m)', color='blue', linestyle='--')
ax3.plot(sol_long.t, sol_long.y[2], label=f'Long Cable (L={L_long}m)', color='teal')
ax3.set_title('Sensitivity Analysis: Effect of Cable Length on Oscillation Frequency')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Amplitude [rad]')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()
