"""Audit: Tepper-García (2006) Voigt approximation — domain-of-validity check
against the [src/models/nerf.py] production code, including a first-principles
computation of the dimensionless damping parameter `a` to test the constant
chosen at nerf.py:222 (`a = 4.7e-4 / b`).

Discharges [COSMO] out-of-scope flag carried forward from [D-54] gate-4a panel
(2026-05-14). Project-completion blocker per session 2026-05-16 audit.

Outputs:
- First-principles `a` at T = {10^3, 10^4, 10^5} K
- Code-computed `a` at the same T (verifying the formula at nerf.py:222)
- Max relative error of tepper_garcia_voigt vs scipy.special.wofz over the
  (a, x) grid the production pipeline actually accesses
- Verdict on Tepper-García validity in our regime
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import wofz  # exact Faddeeva function: w(z) = exp(-z^2) * erfc(-iz)
import torch
import sys

sys.path.insert(0, ".")
from src.models.nerf import tepper_garcia_voigt

# ------------------------------------------------------------------------------
# Part 1: First-principles `a` at three reference temperatures
# ------------------------------------------------------------------------------

# Lyα line constants (NIST / standard atomic data)
GAMMA_LYA = 6.2649e8          # s^-1  (natural line width, Einstein A_21)
LAMBDA_LYA_CM = 1.21567e-5    # cm    (rest wavelength)
NU_0 = 2.4661e15              # Hz    (rest frequency = c / lambda)
C_KM_S = 299792.458           # km/s  (speed of light)
K_B = 1.380649e-23            # J/K
M_H = 1.6735575e-27           # kg    (atomic hydrogen mass)

def b_thermal_kms(T_K: float) -> float:
    """Thermal Doppler parameter (pure thermal, no turbulence)."""
    b_m_s = math.sqrt(2 * K_B * T_K / M_H)
    return b_m_s / 1000.0  # m/s -> km/s

def a_first_principles(T_K: float) -> float:
    """Standard Voigt damping parameter: a = Gamma / (4*pi*Delta_nu_b)
    where Delta_nu_b = nu_0 * b / c."""
    b_kms = b_thermal_kms(T_K)
    delta_nu_b = NU_0 * (b_kms / C_KM_S)  # Hz
    return GAMMA_LYA / (4 * math.pi * delta_nu_b)

def a_production_code(T_K: float) -> float:
    """Reproduces nerf.py:221-222 verbatim."""
    b = 12.85 * math.sqrt(T_K / 10000.0)
    return 4.7e-4 / b

print("=" * 72)
print("Part 1: First-principles vs production-code `a` at three temperatures")
print("=" * 72)
print(f"{'T [K]':<10}{'b_thermal':<14}{'a_first_principles':<22}{'a_production':<16}{'ratio':<10}")
for T in [1e3, 1e4, 1e5]:
    a_fp = a_first_principles(T)
    a_pc = a_production_code(T)
    b = b_thermal_kms(T)
    ratio = a_fp / a_pc
    print(f"{T:<10.0e}{b:<14.3f}{a_fp:<22.4e}{a_pc:<16.4e}{ratio:<10.3f}")

print()

# At T = 10^4 K, the canonical Lya reference:
T_ref = 1e4
a_fp_ref = a_first_principles(T_ref)
a_pc_ref = a_production_code(T_ref)
print(f"At T = 10^4 K reference: a_first_principles = {a_fp_ref:.4e}")
print(f"                         a_production_code  = {a_pc_ref:.4e}")
print(f"                         ratio              = {a_fp_ref / a_pc_ref:.3f}")
print()

# ------------------------------------------------------------------------------
# Part 2: Numerical comparison of TG2006 vs scipy.special.wofz
# ------------------------------------------------------------------------------

print("=" * 72)
print("Part 2: TG2006 vs scipy.special.wofz over the (a, x) grid accessed")
print("=" * 72)
print()
print("The Hjerting function H(a, x) = Re[wofz(x + i*a)].")
print()

def hjerting_exact(a: float, x: np.ndarray) -> np.ndarray:
    """Exact Hjerting function via the Faddeeva function."""
    return np.real(wofz(x + 1j * a))

# Test two grids:
# A) Using the production-code `a` values (12.85x smaller than first-principles)
# B) Using the first-principles `a` values

x_grid = np.linspace(-10, 10, 2001)

def compare_at(a: float, label: str) -> None:
    a_t = torch.tensor(a, dtype=torch.float64)
    x_t = torch.tensor(x_grid, dtype=torch.float64)
    H_tg = tepper_garcia_voigt(a_t, x_t).numpy()
    H_ex = hjerting_exact(a, x_grid)
    diff = H_tg - H_ex
    abs_err = np.abs(diff)
    # Relative error only where exact is not vanishingly small
    rel_err = np.where(np.abs(H_ex) > 1e-12, abs_err / np.abs(H_ex), 0.0)
    print(f"  {label}: a = {a:.4e}")
    print(f"    max abs err over |x|<=10:       {abs_err.max():.3e}")
    print(f"    max abs err in core   |x|<=3:   {abs_err[np.abs(x_grid)<=3].max():.3e}")
    print(f"    max abs err in wings  3<|x|<=10:{abs_err[(np.abs(x_grid)>3) & (np.abs(x_grid)<=10)].max():.3e}")
    print(f"    max rel err where |H|>1e-6:     {rel_err[np.abs(H_ex)>1e-6].max():.3e}")
    print(f"    max rel err where |H|>1e-3:     {rel_err[np.abs(H_ex)>1e-3].max():.3e}")
    print()

print("(A) At PRODUCTION-CODE a-values (the 12.85x-smaller-than-physical regime):")
for T in [1e3, 1e4, 1e5]:
    compare_at(a_production_code(T), f"T={T:.0e} K, b={b_thermal_kms(T):.2f} km/s")

print("(B) At FIRST-PRINCIPLES a-values (the physically correct regime):")
for T in [1e3, 1e4, 1e5]:
    compare_at(a_first_principles(T), f"T={T:.0e} K, b={b_thermal_kms(T):.2f} km/s")

# ------------------------------------------------------------------------------
# Part 3: Practical impact on tau profile
# ------------------------------------------------------------------------------

print("=" * 72)
print("Part 3: Practical impact on tau profile at line wings")
print("=" * 72)
print()
print("At a saturated Lya line core (|x|=3..10), the wing strength scales ~ a:")
print(f"  Wing H(a, x=5)   at a={a_production_code(1e4):.2e}: {hjerting_exact(a_production_code(1e4), np.array([5.0]))[0]:.3e}")
print(f"  Wing H(a, x=5)   at a={a_first_principles(1e4):.2e}: {hjerting_exact(a_first_principles(1e4), np.array([5.0]))[0]:.3e}")
print(f"  Wing H(a, x=10)  at a={a_production_code(1e4):.2e}: {hjerting_exact(a_production_code(1e4), np.array([10.0]))[0]:.3e}")
print(f"  Wing H(a, x=10)  at a={a_first_principles(1e4):.2e}: {hjerting_exact(a_first_principles(1e4), np.array([10.0]))[0]:.3e}")
print()
print("Wing strength ratio (first-principles / production):")
for x_val in [3, 5, 10]:
    r = hjerting_exact(a_first_principles(1e4), np.array([float(x_val)]))[0] / \
        hjerting_exact(a_production_code(1e4), np.array([float(x_val)]))[0]
    print(f"  x = {x_val}: ratio = {r:.2f}")
print()
print("[D-24] DLA mask threshold (per LEDGER §3 [D-24]): τ_max=10 cap; DLAs (τ>10)")
print("are masked. Unsaturated Lya forest (τ<<1) has wing-residual <<1 in either")
print("regime, so the wing-strength discrepancy primarily affects saturated systems")
print("on the path to the DLA mask cap.")
