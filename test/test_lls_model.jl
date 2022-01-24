function lls_test()
  @testset "Test with matrices" begin
    for A in [Matrix(1.0I, 10, 3) .+ 1, sparse(1.0I, 10, 3) .+ 1],
      C in [ones(1, 3), [ones(1, 3); -I], sparse(ones(1, 3))]

      b = collect(1.0:10.0)
      nequ, nvar = size(A)
      ncon = size(C, 1)
      nls = LLSModel(A, b, C = C, lcon = zeros(ncon), ucon = zeros(ncon))
      x = [1.0; -1.0; 1.0]

      @test isapprox(A * x - b, residual(nls, x), rtol = 1e-8)
      @test A == jac_residual(nls, x)
      I, J = jac_structure_residual(nls)
      V = jac_coord_residual(nls, x)
      @test A == sparse(I, J, V, nequ, nvar)
      I, J = hess_structure_residual(nls)
      V = hess_coord_residual(nls, x, ones(nequ))
      @test sparse(I, J, V, nvar, nvar) == zeros(nvar, nvar)
      @test hess_residual(nls, x, ones(nequ)) == zeros(nvar, nvar)
      for i = 1:nequ
        @test isapprox(zeros(nvar, nvar), jth_hess_residual(nls, x, i), rtol = 1e-8)
      end

      I, J = jac_structure(nls)
      V = jac_coord(nls, x)
      @test sparse(I, J, V, ncon, nvar) == C

      @test nls.meta.nlin == length(nls.meta.lin) == ncon
      @test nls.meta.nnln == length(nls.meta.nln) == 0
    end
    Arows, Acols, Avals = findnz(sparse(1.0I, 10, 3) .+ 1)
    A = sparse(1.0I, 10, 3) .+ 1
    Crows, Ccols, Cvals = findnz(sparse(ones(1, 3)))
    C = ones(1, 3)
    b = collect(1.0:10.0)
    nequ, nvar = size(A)
    ncon = size(C, 1)
    nls = LLSModel(
      Arows,
      Acols,
      Avals,
      nvar,
      b,
      Crows = Crows,
      Ccols = Ccols,
      Cvals = Cvals,
      lcon = zeros(ncon),
      ucon = zeros(ncon),
    )
    x = [1.0; -1.0; 1.0]
    @test isapprox(A * x - b, residual(nls, x), rtol = 1e-8)
    @test A == jac_residual(nls, x)
    I2, J = jac_structure_residual(nls)
    V = jac_coord_residual(nls, x)
    @test A == sparse(I2, J, V, nequ, nvar)
    I2, J = hess_structure_residual(nls)
    V = hess_coord_residual(nls, x, ones(nequ))
    @test sparse(I2, J, V, nvar, nvar) == zeros(nvar, nvar)
    @test hess_residual(nls, x, ones(nequ)) == zeros(nvar, nvar)
    for i = 1:nequ
      @test isapprox(zeros(nvar, nvar), jth_hess_residual(nls, x, i), rtol = 1e-8)
    end

    I2, J = jac_structure(nls)
    V = jac_coord(nls, x)
    @test sparse(I2, J, V, ncon, nvar) == C

    @test nls.meta.nlin == length(nls.meta.lin) == ncon
    @test nls.meta.nnln == length(nls.meta.nln) == 0
  end

  @testset "Test with LinearOperators" begin
    for A in [LinearOperator(Matrix(1.0I, 10, 3) .+ 1)],
      C in [LinearOperator(ones(1, 3))]

      b = collect(1.0:10.0)
      nequ, nvar = size(A)
      ncon = size(C, 1)
      nls = LLSModel(A, b, C = C, lcon = zeros(ncon), ucon = zeros(ncon))
      x = [1.0; -1.0; 1.0]

      @test isapprox(A * x - b, residual(nls, x), rtol = 1e-8)
      @test A == jac_residual(nls, x)
      @test Matrix(hess_residual(nls, x, ones(nequ))) == zeros(nvar, nvar)
      for i = 1:nequ
        @test isapprox(zeros(nvar, nvar), Matrix(jth_hess_residual(nls, x, i)), rtol = 1e-8)
      end
      @show nls
      @test C == jac(nls, x)

      @test nls.meta.nlin == length(nls.meta.lin) == ncon
      @test nls.meta.nnln == length(nls.meta.nln) == 0
    end
  end
end

lls_test()
