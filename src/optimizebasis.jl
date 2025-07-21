const ATOL = 1e-12    # quadrature accuracy

χ_STO(r, ζ; n, l) = r^(n - 1) * exp(-ζ * r) * r^l
sto_norm(ζ; n, l) = sqrt((2ζ)^(2n + 2l + 1) / factorial(2n + 2l))

χ_G(r, α; l) = r^l * exp(-α * r^2)
radial_integral(f) = QuadGK.quadgk(r -> f(r) * r^2, 0.0, Inf; atol = ATOL)[1]

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

function inner_fit(alphas, ζ; n, l)
    A, B, C = build_matrices(alphas, ζ; n = n, l = l)
    coeffs  = C \ B
    coeffs ./= sqrt(dot(coeffs, C * coeffs))            # renormalise
    rms²    = A - 2dot(coeffs, B) + dot(coeffs, C * coeffs)
    return coeffs, sqrt(rms²)
end

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

function print_gaussian94(io, element, n, l, β, γ, coeffs)
    lchar = "spdfgh"[l + 1]           # 0→s, 1→p …
    k     = length(coeffs)
    println(io, " $element   $lchar   $k")
    for (i, c) in enumerate(coeffs)
        α = β * γ^(i - 1)
        print(io, " $α $c\n")
    end
end

function optimizebasis(molecule, bssettings)
    atoms = getatoms(molecule)

    basis = GaussianBasisSet[]
    
    for atom in atoms
        println("Optimizing basis for atom: ", atom.symbol)
        for (ν, l, k) in bssettings[atom.symbol]
            println("Optimizing for n=$ν, l=$l, k=$k")
            β, γ, ζ, alphas, coeffs, rms = optimise_basis_v2(k, ν, l)
            coeffs =permutedims(vcat(coeffs))
            alphas = permutedims(vcat(alphas))
            print(alphas)
            for momentum in _angularmomentum(l)
                ℓ = momentum[1]
                println(" ℓ = $ℓ")
                m = momentum[2]
                println(" m = $m")
                n = momentum[3]
                println(" n = $n")
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

    return basis
end

