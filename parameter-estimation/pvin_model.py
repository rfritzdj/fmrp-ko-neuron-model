"""
pvin_model.py
=============

Hodgkin-Huxley model of a parvalbumin (PV) interneuron, with seven free
parameters (the maximal conductances + total Ca buffer):

    gNa, gKv1, gKv3, gCa, gSK, gleak, Btot

All kinetics are fixed (parameters Aah, Sah, etc. are not fit). The model
is implemented in Numba for ~50× speedup over pure Python; one 2.5-s
simulation at dt = 0.02 ms takes ~25 ms after JIT warm-up.

Used by every script in this pipeline. The only thing that changes
between scripts is the input current waveform.
"""

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Reversal potentials and biophysical constants (all fixed)
# ---------------------------------------------------------------------------
VNa, VK, VCa = 58.0, -80.0, 68.0
Cm           = 30.0       # membrane capacitance, pF
pgamma       = 0.01       # Ca extrusion rate constant, 1/ms
KD           = 0.1        # Ca buffer dissociation constant, µM
F            = 0.0964853321  # Faraday / 1e6, for unit consistency
mArea        = 3000.0     # membrane area scaling, µm²
d            = 0.1        # submembrane shell depth, µm
Car          = 0.07       # resting Ca, µM

# Sodium channel kinetics
Vm, Sm = -20.0, -7.0
Aah, Sah, Vah = 0.0025, 10.0, 18.4
Abh, Sbh, Vbh = 0.094, -5.5, -31.0

# Kv1 channel kinetics
Aan1, Van1, San1 = 0.002, -36.0, -9.0
Abn1, Vbn1, Sbn1 = 0.017, -36.75, 6.785

# Kv3 channel kinetics
Aan3, Van3, San3 = 3.2, 96.0, -12.6
Abn3, Vbn3, Sbn3 = 0.34, -36.0, 13.965

# Ca channel kinetics
Va, Sa = 3.5, -11.4

# SK channel kinetics
nk, ksk = 5.0, 0.8


@njit(cache=False)
def vtrap(dV, S):
    """Numerically stable Hodgkin-Huxley rate term.

    Returns -dV / (exp(dV/S) - 1), using expm1 for stability near dV = 0.
    The earlier `dV / (1 - exp(x))` form gave divide-by-zero for x near 0;
    expm1 handles this with a Taylor expansion under the hood.
    """
    x = dV / S
    if np.abs(x) < 1e-6:
        return -S * (1.0 - x / 2.0)
    return -dV / np.expm1(x)


@njit(cache=False)
def PVIN_HH_deriv(y, Iapp, gNa, gKv1, gKv3, gCa, gSK, gleak, Bt, Vleak):
    """Right-hand side of the 5-state ODE: dy/dt = f(y, I, params).

    State vector y = [V, h, n1, n3, ca2i]:
      V    : membrane voltage (mV)
      h    : Na inactivation gate (0..1)
      n1   : Kv1 activation gate (0..1)
      n3   : Kv3 activation gate (0..1)
      ca2i : intracellular Ca concentration (µM)
    """
    V, h, n1, n3, ca2i = y[0], y[1], y[2], y[3], y[4]

    # --- Na current (transient, inactivating) ---
    mmax = 1.0 / (1.0 + np.exp((V - Vm) / Sm))   # instantaneous m∞
    ah = Aah * np.exp(-(V - Vah) / Sah)
    bh = Abh * vtrap(V - Vbh, Sbh)
    INa = gNa * (mmax ** 3) * h * (V - VNa)

    # --- Kv1 current (slow delayed-rectifier-like) ---
    an1 = Aan1 * vtrap(V - Van1, San1)
    bn1 = Abn1 * np.exp(-(V - Vbn1) / Sbn1)
    IKv1 = gKv1 * (n1 ** 4) * (V - VK)

    # --- Kv3 current (fast delayed-rectifier, narrow spike repolarization) ---
    an3 = Aan3 * vtrap(V - Van3, San3)
    bn3 = Abn3 * np.exp(-(V - Vbn3) / Sbn3)
    IKv3 = gKv3 * (n3 ** 2) * (V - VK)

    # --- Ca current (instantaneous activation, no inactivation) ---
    amax = 1.0 / (1.0 + np.exp((V - Va) / Sa))
    ICa = gCa * (amax ** 2) * (V - VCa)

    # --- SK current (Ca-activated K, drives slow AHP / adaptation) ---
    k_sk_denom = ksk ** nk + ca2i ** nk
    k_sk = 0.5 if np.abs(k_sk_denom) < 1e-12 else (ca2i ** nk) / k_sk_denom
    ISK = gSK * k_sk * (V - VK)

    # --- Leak ---
    Ileak = gleak * (V - Vleak)

    # --- Voltage equation ---
    dVdt = (-Ileak - INa - IKv1 - IKv3 - ICa - ISK + Iapp) / Cm

    # --- Gate dynamics ---
    dhdt = ah * (1.0 - h) - bh * h

    an1_sum = an1 + bn1
    if an1_sum < 1e-12:
        an1max, tau_n1max = 0.5, 1e12
    else:
        an1max, tau_n1max = an1 / an1_sum, 1.0 / an1_sum
    dn1dt = (an1max - n1) / tau_n1max

    dn3dt = an3 * (1.0 - n3) - bn3 * n3

    # --- Ca dynamics (single-compartment with linear buffer) ---
    # ICa drives [Ca] up; pgamma drives it back to Car. Buffer (Bt/KD) slows both.
    dCadt = (-ICa / (2.0 * F * mArea * d) - pgamma * (ca2i - Car)) / (1.0 + Bt / KD)

    return dVdt, dhdt, dn1dt, dn3dt, dCadt


@njit(cache=False)
def solve_pvin_rk4(y0, tspan, Idata, gNa, gKv1, gKv3, gCa, gSK, gleak, Bt, Vleak):
    """Fixed-step RK4 integration of the PVIN ODE.

    Parameters
    ----------
    y0    : initial state vector (5,)
    tspan : time array in ms, must be uniformly spaced
    Idata : applied current in pA, same length as tspan
    g*, Bt, Vleak : model parameters

    Returns
    -------
    V_out : voltage trace (mV), shape (len(tspan),). Any NaN means the
            simulation diverged at that timestep; following samples are
            also NaN.
    """
    n_steps = len(tspan)
    dt = tspan[1] - tspan[0]
    V_out = np.zeros(n_steps)
    V, h, n1, n3, ca2i = y0[0], y0[1], y0[2], y0[3], y0[4]
    V_out[0] = V
    y_tmp = np.empty(5)

    for i in range(1, n_steps):
        I0, I1 = Idata[i - 1], Idata[i]
        I_half = 0.5 * (I0 + I1)

        # k1
        y_tmp[0]=V; y_tmp[1]=h; y_tmp[2]=n1; y_tmp[3]=n3; y_tmp[4]=ca2i
        k1V,k1h,k1n1,k1n3,k1ca = PVIN_HH_deriv(y_tmp, I0,
            gNa,gKv1,gKv3,gCa,gSK,gleak,Bt,Vleak)

        # k2
        y_tmp[0]=V+0.5*dt*k1V; y_tmp[1]=h+0.5*dt*k1h
        y_tmp[2]=n1+0.5*dt*k1n1; y_tmp[3]=n3+0.5*dt*k1n3
        y_tmp[4]=ca2i+0.5*dt*k1ca
        k2V,k2h,k2n1,k2n3,k2ca = PVIN_HH_deriv(y_tmp, I_half,
            gNa,gKv1,gKv3,gCa,gSK,gleak,Bt,Vleak)

        # k3
        y_tmp[0]=V+0.5*dt*k2V; y_tmp[1]=h+0.5*dt*k2h
        y_tmp[2]=n1+0.5*dt*k2n1; y_tmp[3]=n3+0.5*dt*k2n3
        y_tmp[4]=ca2i+0.5*dt*k2ca
        k3V,k3h,k3n1,k3n3,k3ca = PVIN_HH_deriv(y_tmp, I_half,
            gNa,gKv1,gKv3,gCa,gSK,gleak,Bt,Vleak)

        # k4
        y_tmp[0]=V+dt*k3V; y_tmp[1]=h+dt*k3h
        y_tmp[2]=n1+dt*k3n1; y_tmp[3]=n3+dt*k3n3
        y_tmp[4]=ca2i+dt*k3ca
        k4V,k4h,k4n1,k4n3,k4ca = PVIN_HH_deriv(y_tmp, I1,
            gNa,gKv1,gKv3,gCa,gSK,gleak,Bt,Vleak)

        # Combine
        V    += (dt/6.0)*(k1V  + 2.0*k2V  + 2.0*k3V  + k4V)
        h    += (dt/6.0)*(k1h  + 2.0*k2h  + 2.0*k3h  + k4h)
        n1   += (dt/6.0)*(k1n1 + 2.0*k2n1 + 2.0*k3n1 + k4n1)
        n3   += (dt/6.0)*(k1n3 + 2.0*k2n3 + 2.0*k3n3 + k4n3)
        ca2i += (dt/6.0)*(k1ca + 2.0*k2ca + 2.0*k3ca + k4ca)

        # Bail out on divergence
        if not np.isfinite(V) or np.abs(V) > 1000.0:
            for j in range(i, n_steps):
                V_out[j] = np.nan
            return V_out
        V_out[i] = V
    return V_out


# ---------------------------------------------------------------------------
# Steady-state initial conditions, used to seed the integration so the cell
# starts at rest (h, n1, n3, ca2i in equilibrium given V0)
# ---------------------------------------------------------------------------

@njit(cache=False)
def hmax(V0):
    """Steady-state Na inactivation at voltage V0."""
    a = Aah * np.exp(-(V0 - Vah) / Sah)
    denom = a + (Abh * vtrap(V0 - Vbh, Sbh))
    return 0.5 if denom < 1e-12 else a / denom

@njit(cache=False)
def n1max_fn(V0):
    """Steady-state Kv1 activation at voltage V0."""
    a = Aan1 * vtrap(V0 - Van1, San1)
    denom = a + (Abn1 * np.exp(-(V0 - Vbn1) / Sbn1))
    return 0.5 if denom < 1e-12 else a / denom

@njit(cache=False)
def n3max_fn(V0):
    """Steady-state Kv3 activation at voltage V0."""
    a = Aan3 * vtrap(V0 - Van3, San3)
    denom = a + (Abn3 * np.exp(-(V0 - Vbn3) / Sbn3))
    return 0.5 if denom < 1e-12 else a / denom

@njit(cache=False)
def ca2i0(V0, gCa):
    """Steady-state intracellular Ca at voltage V0.

    Computed by setting dCa/dt = 0 and solving for ca2i, assuming the
    initial Ca current at V0 is balanced by pgamma * (ca2i - Car).
    """
    ICa = gCa * (1.0 / (1.0 + np.exp((V0 - Va) / Sa))) ** 2 * (V0 - VCa)
    return (ICa / (2.0 * F * mArea * d * pgamma)) + Car


def make_y0(V0, gCa):
    """Convenience: build the 5-element initial state vector for a given V0."""
    return np.array([V0,
                     hmax(V0),
                     n1max_fn(V0),
                     n3max_fn(V0),
                     ca2i0(V0, gCa)])


def warm_up_jit(tspan_arr, Idata_arr, Vleak):
    """Call the JIT'd functions once on small arrays so subsequent calls
    don't pay compilation cost. Returns nothing."""
    n = min(200, len(tspan_arr))
    y0 = make_y0(Vleak, 5.0)
    _ = solve_pvin_rk4(y0, tspan_arr[:n], Idata_arr[:n],
                       300., 15., 180., 8., 10., 3., 80., Vleak)
