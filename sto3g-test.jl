###############################################################################
#   STO-nG regression test (H 1s example)                                     #
#   -----------------------------------------------------------------------   #
#   • Uses numerical quadrature (QuadGK) for all radial integrals.            #
#   • Shows two optimisation routes                                           #
#       1.  Direct normal-equation solve  C*c = B  (strict least-squares)     #
#       2.  Gradient descent with automatic differentiation (Zygote)          #
#   • Prints the RMS radial error and the fitted coefficients.                #
#   -----------------------------------------------------------------------   #
#   REQUIRE:  julia> ] add QuadGK Zygote                                      #
###############################################################################

using LinearAlgebra, QuadGK, Zygote

# ----------  1.  Slater-type (target) orbital --------------------------------
"""
    χ_STO(r; ζ=1.24, n=1)

Unnormalised radial part  r^{n-1}·exp(-ζ·r).
"""
χ_STO(r; ζ=1.24, n=1) = r^(n-1) * exp(-ζ * r)

"""
    normalise_STO(ζ; n=1)

Returns the normalisation constant N such that
∫₀^∞ |N·χ_STO(r)|²·r² dr = 1
"""
function normalise_STO(ζ; n=1)
    # analytic for n = 1,2,…  :  N = √( (2ζ)^{2n+1} / (2n)! )
    return sqrt( (2ζ)^(2n+1) / factorial(2n) )
end

# ----------  2.  Primitive Gaussian -----------------------------------------
"""
    χ_G(r, α; l=0)

Primitive Gaussian  r^{l}·exp(-α·r²)
(Angular part is ignored because we only care about the radial fit.)
"""
χ_G(r, α; l=0) = r^l * exp(-α * r^2)

# ----------  3.  Radial integrals (numerical) -------------------------------
const atol = 1e-12       # quad accuracy

# ∫₀^∞ f(r) r² dr  (measure already contains 4π average over angles)
function radial_integral(f)
    value, _err = QuadGK.quadgk(r -> f(r) * r^2, 0.0, Inf; atol)
    return value
end
"""
    build_matrices(alphas, ζ; l=0, n=1)

Return (A, B, C) for the quadratic error
R² = A − 2 cᵀ B + cᵀ C c
"""
function build_matrices(alphas, ζ; l=0, n=1)
    k = length(alphas)
    # Normalised STO
    N = normalise_STO(ζ; n)
    sto = r -> N * χ_STO(r; ζ, n)
    # A  = ⟨STO|STO⟩  = 1 by construction
    A = 1.0
    # Bᵢ = ⟨PGFᵢ|STO⟩
    B = [ radial_integral(r -> χ_G(r, α; l) * sto(r)) for α in alphas ]
    # Cᵢⱼ = ⟨PGFᵢ|PGFⱼ⟩
    C = [ radial_integral(r -> χ_G(r, αi; l) * χ_G(r, αj; l)) 
          for αi in alphas, αj in alphas ]
    return A, B, Symmetric(reshape(C, k, k))
end

# ----------  4.  Direct least-squares solution ------------------------------
function fit_direct(alphas, ζ; l=0, n=1)
    A, B, C = build_matrices(alphas, ζ; l, n)
    c = C \ B              # solve normal equations
    # renormalise contracted Gaussian
    norm = sqrt(dot(c, C * c))
    c ./= norm
    # RMS error
    R2 = A - 2dot(c, B) + dot(c, C * c)
    return c, sqrt(R2)
end

# ----------  5.  Gradient descent (AD) --------------------------------------
"""
    fit_AD(alphas, ζ; l=0, n=1, lr=1e-2, maxiter=2000)

Crude gradient descent on the same quadratic loss, to illustrate that
AD finds the same minimum as the linear solve.
"""
function fit_AD(alphas, ζ; l=0, n=1, lr=1e-2, maxiter=2000)
    k = length(alphas)
    A, B, C = build_matrices(alphas, ζ; l, n)
    c = fill(0.3, k)                   # naive start
    loss(c) = A - 2dot(c, B) + dot(c, C * c)
    for it = 1:maxiter
        grad = Zygote.gradient(loss, c)[1]
        c .-= lr * grad
        if norm(grad) < 1e-10
            break
        end
    end
    # renormalise and RMS
    c ./= sqrt(dot(c, C * c))
    R  = sqrt(loss(c))
    return c, R
end

# ----------  6.  Demo: H 1s STO-3G ------------------------------------------
ζ_H  = 1.24                       # standard value used by STO-3G
αs   = [0.109818, 0.405771, 2.22766]   # H 1s STO-3G exponents

#c  = [0.9817067, 0.949464, 0.2959065]

#H     0
#S    3   1.00
#      0.3425250914D+01       0.1543289673D+00
#      0.6239137298D+00       0.5353281423D+00
#      0.1688554040D+00       0.4446345422D+00
#****

c_direct, RMS_direct = fit_direct(αs, ζ_H)
c_AD,      RMS_AD    = fit_AD(αs, ζ_H; lr=1e-1)

println("=== STO-3G (H 1s) fit with fixed exponents ===")
println("Direct linear solve:")
println("   c  = ", round.(c_direct, sigdigits=7))
println("   RMS = ", print("%.3e", RMS_direct))
println("Automatic-diff AD gradient descent:")
println("   c  = ", round.(c_AD, sigdigits=7))
println("   RMS = ", print("%.3e", RMS_AD))
