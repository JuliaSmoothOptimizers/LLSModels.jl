function lls_interaction_test()
  @testset "interaction_test" begin
    m, n = 20, 10
    A = rand(m, n)
    b = rand(m)
    lcon, ucon = zeros(m), fill(Inf, m)
    C = ones(m, n)
    lvar, uvar = fill(-10.0, n), fill(200.0, n)
    nls = LLSModel(A, b, lvar = lvar, uvar = uvar, C = C, lcon = lcon, ucon = ucon)
    FLLS = FeasibilityFormNLS(nls)
    Hrows, Hcols = hess_structure(FLLS)
    Hvals = hess_coord(FLLS, FLLS.meta.x0)
    @test Hrows == [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    @test Hcols == Hrows
    @test length(Hrows) == length(Hvals)
    @test Hvals == ones(length(Hrows))
  end
end

lls_interaction_test()
