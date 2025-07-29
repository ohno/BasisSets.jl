@time @testset "molecule.jl" begin
    # benchmark for testing
    benchmark = Molecule(["H", "H"], [0.0 0.0 0.0; 0.0 0.0 0.74], [1, 1])
    # test `molecule()`
    mol1 = molecule("./data/hydrogen/h-atom.xyz")
    # test `@molecule {}`
    mol2 = @molecule {
                H  0.0  0.0  0.0
                H  0.0  0.0  0.74
            }
    # test `BasisSets.parse_xyz()`
    mol3 = BasisSets.parse_xyz("
            H  0.0  0.0  0.0
            H  0.0  0.0  0.74
        ")
    # @test for all cases
    for mol in [mol1, mol2, mol3]
        @test mol isa Molecule
        @test mol.atoms == benchmark.atoms
        @test mol.coords == benchmark.coords
        @test mol.numbers == benchmark.numbers
    end
end