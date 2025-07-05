module BasisSets
    using 
        HTTP, 
        JSON3, 
        LinearAlgebra, 
        StaticArrays

    include("periodictable.jl")
    include("molecule.jl")
    include("parser.jl")
    include("getdata.jl")

    export 
        Atom, 
        getatom,
        Molecule,
        molecule,
        getatoms,
        doublefactorial,
        normalization,
        parsebasis,
        parsebasis_fromfile,
        metadata

    const _METADATA = let 
        path = joinpath(@__DIR__, "data", "METADATA.json")  # __DIR__ == src/
        JSON3.read(path)
    end

    metadata() = _METADATA

end