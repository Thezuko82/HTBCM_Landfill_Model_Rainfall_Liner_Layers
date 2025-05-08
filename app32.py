import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("HTBCM Landfill Model with Rainfall and Liner Layers")

# Sidebar Inputs
st.sidebar.header("Landfill & Environmental Parameters")
length = st.sidebar.slider("Landfill Depth (m)", 5, 100, 30)
time_days = st.sidebar.slider("Simulation Time (days)", 10, 365, 100)
dx = 1.0
nx = int(length / dx)
dt = 1.0
t_steps = int(time_days / dt)
x = np.linspace(0, length, nx)
t = np.linspace(0, time_days, t_steps)
X, T = np.meshgrid(x, t)

# Leachate transport
velocity = st.sidebar.slider("Leachate Velocity (m/day)", 0.001, 1.0, 0.05)
dispersion = st.sidebar.slider("Dispersion Coefficient (m2/day)", 0.001, 1.0, 0.01)

# Rainfall
st.sidebar.subheader("Rainfall Infiltration")
rainfall_rate = st.sidebar.slider("Daily Rainfall (mm/day)", 0.0, 100.0, 10.0)
infiltration_coeff = st.sidebar.slider("Infiltration Coefficient", 0.0, 1.0, 0.5)
rain_input = (rainfall_rate / 1000.0) * infiltration_coeff  # m/day

# Liner
st.sidebar.subheader("Liner Properties")
liner_thickness = st.sidebar.slider("Liner Thickness (m)", 0.1, 5.0, 1.0)
liner_perm = st.sidebar.slider("Liner Permeability (m/s)", 1e-10, 1e-6, 1e-9, format="%.1e")
liner_leakage = liner_perm * (1.0 / liner_thickness) * dt * 86400  # simple Darcy estimate

# Sorption
sorption_kd = st.sidebar.slider("Sorption Coefficient Kd (L/kg)", 0.0, 10.0, 1.0)

# Initial concentration & biomass
initial_conc = st.sidebar.number_input("Initial Concentration (mg/L)", 0.0, 1000.0, 100.0)
initial_biomass = st.sidebar.number_input("Initial Biomass (mg/L)", 1.0, 500.0, 50.0)

# Biodegradation
mu_max = st.sidebar.slider("μmax (1/day)", 0.001, 1.0, 0.1)
Ks = st.sidebar.slider("Half-saturation Ks (mg/L)", 0.1, 500.0, 50.0)
biogas_yield = st.sidebar.slider("Biogas Yield (L/g COD)", 0.1, 1.0, 0.5)

# Initialize matrices
C = np.zeros((t_steps, nx))
B = np.zeros((t_steps, nx))
C[0, :] = initial_conc
B[0, :] = initial_biomass
gas = np.zeros(t_steps)

# Monod kinetics
def monod(C, B, mu_max, Ks):
    return mu_max * C / (Ks + C) * B

# Main simulation loop
for n in range(1, t_steps):
    dCdx = np.gradient(C[n-1, :], dx)
    d2Cdx2 = np.gradient(dCdx, dx)
    
    # Infiltration at surface
    infiltration_input = np.zeros(nx)
    infiltration_input[0] = rain_input * 1000  # mg/L per day

    # Liner leakage at bottom
    liner_loss = np.zeros(nx)
    liner_loss[-1] = liner_leakage * C[n-1, -1]

    # Sorbed fraction
    C_mobile = C[n-1, :] / (1 + sorption_kd)

    growth = monod(C_mobile, B[n-1, :], mu_max, Ks)

    C[n, :] = C[n-1, :] - velocity * dCdx * dt + dispersion * d2Cdx2 * dt - growth * dt
    C[n, :] += infiltration_input * dt
    C[n, :] -= liner_loss
    C[n, :] = np.maximum(C[n, :], 0)

    B[n, :] = B[n-1, :] + growth * dt
    gas[n] = gas[n-1] + np.sum(growth * dt * biogas_yield)

# Display concentration profile
st.subheader("Leachate Concentration (mg/L)")
fig1, ax1 = plt.subplots()
cf = ax1.contourf(x, t, C, cmap="viridis", levels=30)
fig1.colorbar(cf, ax=ax1)
ax1.set_xlabel("Depth (m)")
ax1.set_ylabel("Time (days)")
ax1.set_title("2D Leachate Concentration")
st.pyplot(fig1)

# Biogas plot
st.subheader("Cumulative Biogas Production")
fig2, ax2 = plt.subplots()
ax2.plot(t, gas, color='green')
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Cumulative Biogas (L)")
st.pyplot(fig2)

st.markdown("""
✅ Included Features:
- Rainfall infiltration affects surface concentration.
- Liner permeability controls bottom losses.
- Sorption and Monod-based degradation.
- 2D visualization of depth-time leachate profile.
""")
