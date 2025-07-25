###############################################################################
# compare_gauss_vs_slater.jl                                                  #
#   julia> ] add QuadGK BenchmarkTools                                        #
###############################################################################

using QuadGK              # numerical ∫
using BenchmarkTools      # stable timing

# 1. analytic overlap of two s‑type primitive Gaussians (centres coincide)
#    S_G  =  (π / (α + β))^(3/2)
gauss_overlap(α, β) = (π / (α + β))^(3/2)

# 2. numerical overlap of two Slater 1s functions (centres coincide)
#    Normalised STO: χ(r) = (ζ^3/π)^{1/2} * exp(-ζ r)
function slater_overlap(ζ1, ζ2)
    pref = (ζ1^3 * ζ2^3)^(1/2) / π
    integrand(r) = 4π * pref * exp(-(ζ1 + ζ2) * r) * r^2
    QuadGK.quadgk(integrand, 0.0, Inf; atol = 1e-12, rtol = 1e-12)[1]
end

# --- demo parameters ---------------------------------------------------------
α  = 1.24     # Gaussian exponent
β  = 0.75
ζ1 = 1.24     # Slater effective charges
ζ2 = 0.75

println("Single‑shot results:")
println("  Gaussian overlap  = ", gauss_overlap(α, β))
println("  Slater  overlap   = ", slater_overlap(ζ1, ζ2))

println("\nTiming (100 000 evaluations each):")
@btime gauss_overlap($α, $β)        setup=() evals=100000
@btime slater_overlap($ζ1, $ζ2)     setup=() evals=100000
