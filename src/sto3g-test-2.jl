################################################################################
#  STO‑nG outer optimisation of (β, γ, ζ) with inner LSQ for the coefficients  #
#  --------------------------------------------------------------------------  #
#  • k primitives, even‑tempered: αᵢ = β γ^{i-1}                                #
#  • One STO target  χ_STO(r; ζ, n, l)                                         #
#  • Objective: minimise RMS radial error                                      #
#  • Optimiser:  Fminbox{NelderMead} (bounds enforce β>0, γ>1, ζ>0)            #
################################################################################

using LinearAlgebra
using QuadGK               # numerical radial integrals
using Optim                # derivative‑free optimiser with box constraints
import Optim: converged

# ----------  numerical settings ---------------------------------------------
const atol = 1e-12   # absolute accuracy for quadgk

# ----------  STO target ------------------------------------------------------
χ_STO(r, ζ; n=1, l=0) = r^(n-1) * exp(-ζ*r) * r^l      # radial × angular factor

# analytic normalisation  N = √[(2ζ)^{2n+2l+1}/(2n+2l)!]
function sto_norm(ζ; n, l)
    return sqrt( (2ζ)^(2n+2l+1) / factorial(2n+2l) )
end

# ----------  primitive Gaussian ---------------------------------------------
χ_G(r, α; l=0) = r^l * exp(-α * r^2)

# ----------  helper: radial integral ----------------------------------------
radial_integral(f) = QuadGK.quadgk(r -> f(r) * r^2, 0.0, Inf; atol)[1]

# ----------  build matrices for given αs and ζ ------------------------------
function build_matrices(alphas, ζ; n=1, l=0)
    k = length(alphas)
    Nsto = sto_norm(ζ; n, l)
    sto  = r -> Nsto * χ_STO(r, ζ; n, l)

    # A = <STO|STO> = 1 (normalised), but keep it explicit for completeness
    A = 1.0

    # Bᵢ = <PGFᵢ|STO>
    B = [ radial_integral(r -> χ_G(r, α; l) * sto(r)) for α in alphas ]

    # Cᵢⱼ = <PGFᵢ|PGFⱼ>
    C = [ radial_integral(r -> χ_G(r, αi; l) * χ_G(r, αj; l))
          for αi in alphas, αj in alphas ]
    return A, B, Symmetric(reshape(C, k, k))
end

# ----------  inner LSQ: coeffs + RMS ----------------------------------------
function inner_fit(alphas, ζ; n=1, l=0)
    A, B, C = build_matrices(alphas, ζ; n, l)
    coeffs  = C \ B                          # normal equations
    coeffs ./= sqrt(dot(coeffs, C * coeffs)) # renormalise contracted AO
    RMS²    = A - 2dot(coeffs,B) + dot(coeffs, C * coeffs)
    return coeffs, sqrt(RMS²)
end

# ----------  outer objective -------------------------------------------------
"""
    rms_objective(p, k; n, l)

`p = [β, γ, ζ]`  → returns RMS radial error for k primitives.
"""
function rms_objective(p, k; n=1, l=0)
    β, γ, ζ = p
    αs = [ β * γ^(i-1) for i = 1:k ]
    _, rms = inner_fit(αs, ζ; n, l)
    return rms
end

# ----------  optimisation driver --------------------------------------------
"""
    optimise_basis(k; n=1, l=0, β0=0.15, γ0=3.2, ζ0=1.0)

Returns (β★, γ★, ζ★, coeffs, RMS★).
"""
function optimise_basis(k; n=1, l=0, β0=0.15, γ0=3.2, ζ0=1.0)
    # initial parameter vector
    p0   = [β0, γ0, ζ0]

    # parameter bounds  (β>0, γ>1, ζ>0) – feel free to widen
    lower = [1e-4, 1.01, 0.1]
    upper = [1e2 , 10.0,  5.0]

    obj(p) = rms_objective(p, k; n, l)

    res = optimize(obj, lower, upper, p0,
                   Fminbox{NelderMead}();
                   x_tol = 1e-10, f_tol = 1e-12)

    if !converged(res)
        error("Optimisation failed!")
    end

    β★, γ★, ζ★ = Optim.minimizer(res)
    αs  = [ β★ * γ★^(i-1) for i = 1:k ]
    coeffs, RMS★ = inner_fit(αs, ζ★; n, l)
    return β★, γ★, ζ★, coeffs, RMS★, res
end

# ----------  example: H 1s STO‑3G (k=3, n=1, l=0) ---------------------------
β, γ, ζ, c, rms, res = optimise_basis(3; n=1, l=0)

println("=== STO‑3G automatic fit (H 1s) ===")
println("β       = ", round(β, 8))
println("γ       = ", round(γ, 8))
println("ζ       = ", round(ζ, 8))
println("αᵢ      = [", join(round.( [β*γ^(i-1) for i=1:3], 8 ), ", "), "]")
println("coeffs  = ", round.(c, sigdigits = 7))
println("RMS     = ", @sprintf("%.3e", rms))
println("\nOptim summary:")
println(res)
