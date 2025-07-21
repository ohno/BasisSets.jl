###############################################################################
#  basis_forge.jl                                                             #
#  ------------------------------------------------------------------------   #
#  Build custom Gaussian‑format basis sets via automatic least‑squares fits   #
#  to Slater targets.                                                         #
#                                                                             #
#  Dependencies (install once):                                               #
#      ] add QuadGK Optim                                                     #
###############################################################################

using LinearAlgebra
using QuadGK          # ∫₀^∞ … r² dr integrals
using Optim           # derivative‑free optimisation

# ---------- numerical settings ----------------------------------------------
const ATOL = 1e-12    # quadrature accuracy

# ---------- Slater target ----------------------------------------------------
χ_STO(r, ζ; n, l) = r^(n - 1) * exp(-ζ * r) * r^l
sto_norm(ζ; n, l) = sqrt((2ζ)^(2n + 2l + 1) / factorial(2n + 2l))

# ---------- primitive Gaussian ----------------------------------------------
χ_G(r, α; l) = r^l * exp(-α * r^2)
radial_integral(f) = QuadGK.quadgk(r -> f(r) * r^2, 0.0, Inf; atol = ATOL)[1]

# ---------- build A, B, C matrices ------------------------------------------
function build_matrices(alphas, ζ; n, l)
    k      = length(alphas)
    Nsto   = sto_norm(ζ; n = n, l = l)
    sto(r) = Nsto * χ_STO(r, ζ; n = n, l = l)

    A = 1.0                                             # ⟨STO|STO⟩
    B = [radial_integral(r -> χ_G(r, α; l) * sto(r)) for α in alphas]
    C = [radial_integral(r -> χ_G(r, αi; l) * χ_G(r, αj; l))
         for αi in alphas, αj in alphas]
    return A, B, Symmetric(reshape(C, k, k))
end

# ---------- inner LSQ for coefficients + RMS --------------------------------
function inner_fit(alphas, ζ; n, l)
    A, B, C = build_matrices(alphas, ζ; n = n, l = l)
    coeffs  = C \ B
    coeffs ./= sqrt(dot(coeffs, C * coeffs))            # renormalise
    rms²    = A - 2dot(coeffs, B) + dot(coeffs, C * coeffs)
    return coeffs, sqrt(rms²)
end

# ---------- outer optimisation of β, γ, ζ  (even‑tempered αᵢ) ---------------
function optimise_basis(k, n, l, β0 = 0.15, γ0 = 3.2, ζ0 = 1.0)
    obj(p) = begin
        β, γ, ζ = p
        alphas  = [β * γ^(i - 1) for i = 1:k]
        _, rms  = inner_fit(alphas, ζ; n = n, l = l)
        rms
    end

    p0    = [β0, γ0, ζ0]
    lower = [1e-4, 1.01, 0.1]    # β>0, γ>1, ζ>0
    upper = [1e2,  10.0, 5.0]

    opts  = Optim.Options(x_abstol = 1e-10, f_abstol = 1e-12)
    res   = optimize(obj, lower, upper, p0, Fminbox(NelderMead()), opts)

    β, γ, ζ   = Optim.minimizer(res)
    alphas    = [β * γ^(i - 1) for i = 1:k]
    coeffs, rms = inner_fit(alphas, ζ; n = n, l = l)
    return β, γ, ζ, coeffs, rms
end

function optimise_basis_v2(k, n, l, β0 = 0.15, γ0 = 3.2, ζ0 = 1.0)
    obj(p) = begin
        β, γ, ζ = p
        alphas  = [β * γ^(i - 1) for i = 1:k]
        _, rms  = inner_fit(alphas, ζ; n = n, l = l)
        rms
    end

    p0    = [β0, γ0, ζ0]
    lower = [1e-4, 1.01, 0.1]    # β>0, γ>1, ζ>0
    upper = [1e2,  10.0, 5.0]

    opts  = Optim.Options(x_abstol = 1e-10, f_abstol = 1e-12)
    res   = optimize(obj, lower, upper, p0, Fminbox(NelderMead()), opts)

    β, γ, ζ   = Optim.minimizer(res)
    alphas    = [β * γ^(i - 1) for i = 1:k]
    coeffs, rms = inner_fit(alphas, ζ; n = n, l = l)
    return β, γ, ζ, alphas, coeffs, rms
end

# ---------- helper: print in Gaussian94 block format ------------------------
function print_gaussian94(io, element, n, l, β, γ, coeffs)
    lchar = "spdfgh"[l + 1]           # 0→s, 1→p …
    k     = length(coeffs)
    println(io, " $element   $lchar   $k")
    for (i, c) in enumerate(coeffs)
        α = β * γ^(i - 1)
        print(io, " $α $c\n")
    end
end

# ---------- edit here: elements & shells to build ---------------------------
elements = ["H", "C"]

#  shell_table[element] = list of (n, l, k)
shell_table = Dict(
    "H" => [(1, 0, 6)],                          # 1s STO‑3G
    "C" => [(1, 0, 3),                           # 1s
            (2, 0, 3),                           # 2s
            (2, 1, 3),                           # 2p (shares β,γ with 2s)
            (0, 2, 1)]                           # 3d polarisation
)


β, γ, ζ, alphas, coeffs, rms = optimise_basis_v2(6, 1, 0)
print(coeffs)
println(alphas)
println("β = $β, γ = $γ, ζ = $ζ\n")

function optimizebasis(molecule, settings)
    atoms = getatoms(molecule)

    basis = GaussianBasisSet[]
    
    for atom in atoms
        for setting in settings
            for (idx, (n, l, k)) in enumerate(setting[atom.symbol])
                β, γ, ζ, alphas, coeffs, rms = optimise_basis_v2(k, n, l)
                for momentum in _angularmomentum(l)
                    ℓ = momentum[1]
                    m = momentum[2]
                    n = momentum[3]
                    push!(basis,
                        GaussianBasisSet(
                            atom.coords,
                            alphas,
                            coeffs,
                            normalization.(alphas, ℓ, m, n),
                            length(alphas),
                            ℓ,
                            m,
                            n
                        )
                    )
                end
            end
        end
    end

    return basis
end

function optimizebasis(molecule, settings)
    atoms = getatoms(molecule)
    basis = GaussianBasisSet[]
    for atom in atoms
        for setting in settings
            for (idx, (n, l, k)) in enumerate(setting[atom.symbol])
                β, γ, ζ, alphas, coeffs, rms = optimise_basis_v2(k, n, l)
                for momentum in _angularmomentum(l)
                    ℓ = momentum[1]
                    m = momentum[2]
                    n = momentum[3]
                    push!(basis,
                        GaussianBasisSet(
                            atom.coords,
                            alphas,
                            coeffs,
                            normalization.(alphas, ℓ, m, n),
                            length(alphas),
                            ℓ,
                            m,
                            n
                        )
                    )
                end
            end
        end
    end
    return basis
end

open("my-basis.gbs", "w") do io
    for el in elements
        for (idx, (n, l, k)) in enumerate(shell_table[el])
            # share β, γ between 2s and 2p (SP trick) if consecutive in list
            if el == "C" && n == 2 && l == 1 && idx > 1
                β, γ, _, _, _ = optimise_basis(k, 2, 0)
                ζ = 1.0                          # ζ irrelevant for pure Gaussian fit
                coeffs, _ = inner_fit([β * γ^(i - 1) for i = 1:k],
                                       ζ; n = 2, l = 1)
                print_gaussian94(io, el, n, l, β, γ, coeffs)
            else
                β, γ, ζ, coeffs, _ = optimise_basis(k, n, l)
                print_gaussian94(io, el, n, l, β, γ, coeffs)
            end
        end
    end
end

println("\nBasis‑set file  →  my‑basis.gbs   (Gaussian94 / ORCA %basis ready)")
######################################################################## EOF ###
